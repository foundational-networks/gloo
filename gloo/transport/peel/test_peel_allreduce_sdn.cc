// Standalone end-to-end SDN/Reno numerical test. No PyTorch or Redis required.
// Usage: test_peel_allreduce_sdn RANK IP0,IP1,... TOPOLOGY IFACE BASE_PORT
//        TILE_BYTES UNIQUE_SHARED_BARRIER_DIR [TIMEOUT_MS [REDUCE_PARALLEL=0|1]]
// The barrier directory must be new for every launch. All hosts execute the
// same program/arguments except RANK. World size is inferred from the IP list.

#include "gloo/transport/peel/peel_allreduce.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using gloo::transport::peel::PeelAllreduce;
using gloo::transport::peel::PeelContextConfig;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

enum class Operation { Sum, Product, Min, Max, Band, Bor, Bxor };

const char* name(Operation op) {
  switch (op) {
    case Operation::Sum: return "SUM";
    case Operation::Product: return "PRODUCT";
    case Operation::Min: return "MIN";
    case Operation::Max: return "MAX";
    case Operation::Band: return "BAND";
    case Operation::Bor: return "BOR";
    case Operation::Bxor: return "BXOR";
  }
  return "invalid";
}

template <typename T>
T inputValue(int rank, size_t index, Operation op) {
  if (op == Operation::Product) {
    return static_cast<T>(1 + ((rank + static_cast<int>(index % 3)) % 2));
  }
  if constexpr (std::is_integral<T>::value) {
    return static_cast<T>((rank * 17 + static_cast<int>(index % 31) * 3) % 256);
  } else {
    return static_cast<T>(rank + 1) +
        static_cast<T>(static_cast<int>(index % 17) - 8) / static_cast<T>(4);
  }
}

template <typename T>
T binary(T left, T right, Operation op) {
  switch (op) {
    case Operation::Sum: return left + right;
    case Operation::Product: return left * right;
    case Operation::Min: return std::min(left, right);
    case Operation::Max: return std::max(left, right);
    case Operation::Band:
      if constexpr (std::is_integral<T>::value) { return left & right; }
      break;
    case Operation::Bor:
      if constexpr (std::is_integral<T>::value) { return left | right; }
      break;
    case Operation::Bxor:
      if constexpr (std::is_integral<T>::value) { return left ^ right; }
      break;
  }
  throw std::runtime_error("unsupported test operation");
}

template <typename T>
T expectedValue(int world, size_t index, Operation op) {
  T result = inputValue<T>(0, index, op);
  for (int rank = 1; rank < world; ++rank) {
    result = binary(result, inputValue<T>(rank, index, op), op);
  }
  return result;
}

template <typename T>
bool same(T actual, T expected) {
  if constexpr (std::is_integral<T>::value) {
    return actual == expected;
  } else {
    return std::isfinite(actual) &&
        std::abs(actual - expected) <=
            static_cast<T>(1e-6) * std::max(static_cast<T>(1), std::abs(expected));
  }
}

template <typename T>
PeelAllreduce::ReduceFunction callback(Operation op) {
  return [op](void* output, const void* first, const void* second, size_t count) {
    auto* destination = static_cast<T*>(output);
    const auto* left = static_cast<const T*>(first);
    const auto* right = static_cast<const T*>(second);
    for (size_t i = 0; i < count; ++i) {
      destination[i] = binary(left[i], right[i], op);
    }
  };
}

bool setupBarrier(const fs::path& directory, int rank, int world, int timeout) {
  if (world == 1 && directory == "-") {
    return true;
  }
  fs::create_directories(directory);
  const auto temporary = directory / (std::to_string(rank) + ".tmp");
  const auto ready = directory / (std::to_string(rank) + ".ready");
  {
    std::ofstream marker(temporary);
    if (!marker) { return false; }
    marker << "ready\n";
  }
  fs::rename(temporary, ready);
  const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
  while (Clock::now() < deadline) {
    bool complete = true;
    for (int peer = 0; peer < world; ++peer) {
      complete = complete && fs::exists(directory / (std::to_string(peer) + ".ready"));
    }
    if (complete) { return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  std::cerr << "setup barrier timeout\n";
  return false;
}

template <typename T>
bool allreduceCase(PeelAllreduce& collective, int rank, int world,
                   size_t count, Operation op) {
  std::vector<T> input(std::max<size_t>(count, 1));
  for (size_t i = 0; i < count; ++i) {
    input[i] = inputValue<T>(rank, i, op);
  }
  if (!collective.run(input.data(), count, sizeof(T), callback<T>(op))) {
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    const T expected = expectedValue<T>(world, i, op);
    if (!same(input[i], expected)) {
      std::cerr << "allreduce mismatch rank=" << rank << " op=" << name(op)
                << " count=" << count << " i=" << i << " actual=" << input[i]
                << " expected=" << expected << "\n";
      return false;
    }
  }
  const auto& stats = collective.lastStats();
  if (world > 1 && count > 0 &&
      (stats.reduceScatterBroadcasts == 0 || stats.allgatherBroadcasts == 0)) {
    std::cerr << "nonempty collective omitted an SDN phase\n";
    return false;
  }
  std::cout << "PASS allreduce rank=" << rank << " dtype_bytes=" << sizeof(T)
            << " op=" << name(op) << " elements=" << count
            << " rs_s=" << stats.reduceScatterSeconds
            << " ag_s=" << stats.allgatherSeconds
            << " logical_rs_send_bytes=" << stats.reduceScatterSendBytes
            << " logical_ag_send_bytes=" << stats.allgatherSendBytes << "\n";
  return true;
}

template <typename T>
bool scatterCase(PeelAllreduce& collective, int rank, int world,
                 size_t count, bool alias) {
  const Operation op = Operation::Sum;
  std::vector<T> input(std::max<size_t>(count, 1));
  for (size_t i = 0; i < count; ++i) {
    input[i] = inputValue<T>(rank, i, op);
  }
  const auto original = input;
  const size_t shard = count / world + (count % world != 0);
  const size_t start = std::min(count, static_cast<size_t>(rank) * shard);
  const size_t localCount = collective.localShardElements(count);
  const T sentinel = static_cast<T>(-12345);
  std::vector<T> output(localCount + 1, sentinel);
  T* destination = alias ? input.data() + start : output.data();
  if (!collective.reduceScatter(input.data(), destination, count, sizeof(T), callback<T>(op))) {
    return false;
  }
  for (size_t i = 0; i < localCount; ++i) {
    if (!same(destination[i], expectedValue<T>(world, start + i, op))) {
      std::cerr << "reduce-scatter mismatch rank=" << rank << " alias=" << alias
                << " i=" << i << "\n";
      return false;
    }
  }
  if (output.back() != sentinel) {
    std::cerr << "reduce-scatter wrote beyond valid output\n";
    return false;
  }
  for (size_t i = 0; i < count; ++i) {
    if ((!alias || i < start || i >= start + localCount) && input[i] != original[i]) {
      std::cerr << "reduce-scatter modified an unowned input\n";
      return false;
    }
  }
  if (collective.lastStats().allgatherBroadcasts != 0) {
    std::cerr << "direct reduce-scatter unexpectedly ran allgather\n";
    return false;
  }
  std::cout << "PASS reduce-scatter rank=" << rank << " dtype_bytes=" << sizeof(T)
            << " elements=" << count << " alias=" << alias
            << " local_elements=" << localCount << "\n";
  return true;
}

int main(int argc, char** argv) {
  try {
    if (argc < 8 || argc > 10) {
      std::cerr << "Usage: test_peel_allreduce_sdn RANK IP0,IP1,... TOPOLOGY IFACE "
                   "BASE_PORT TILE_BYTES UNIQUE_SHARED_BARRIER_DIR [TIMEOUT_MS [REDUCE_PARALLEL=0|1]]\n";
      return 2;
    }
    PeelContextConfig config;
    config.rank = std::stoi(argv[1]);
    std::stringstream addresses(argv[2]);
    std::string address;
    int rank = 0;
    while (std::getline(addresses, address, ',')) {
      config.peer_ips[rank++] = address;
    }
    config.world_size = rank;
    config.topology_file = argv[3];
    config.iface_name = argv[4];
    const int base = std::stoi(argv[5]);
    if (base < 1 || base > 65535) { return 2; }
    config.base_port = static_cast<uint16_t>(base);
    config.timeout_ms = argc >= 9 ? std::stoi(argv[8]) : 120000;
    bool reduceParallel = false;
    if (argc == 10) {
      const std::string mode(argv[9]);
      if (mode != "0" && mode != "1") { return 2; }
      reduceParallel = mode == "1";
    }
    config.rto_ms = 500;
    const size_t tileBytes = std::stoull(argv[6]);
    if (tileBytes < sizeof(int64_t)) { return 2; }
    PeelAllreduce collective(config, tileBytes);
    collective.setReduceGroupsParallel(reduceParallel);
    if (!collective.initReno() ||
        !setupBarrier(argv[7], config.rank, config.world_size, config.timeout_ms)) {
      return 3;
    }
    const std::vector<size_t> sizes{
        0, 1, static_cast<size_t>(std::max(0, config.world_size - 1)),
        static_cast<size_t>(config.world_size + 3),
        static_cast<size_t>(config.world_size) * (tileBytes / sizeof(float) + 3) + 1,
        1};
    for (auto op : {Operation::Sum, Operation::Product, Operation::Min, Operation::Max}) {
      for (size_t count : sizes) {
        if (!allreduceCase<float>(collective, config.rank, config.world_size, count, op)) {
          return 4;
        }
      }
    }
    for (auto op : {Operation::Sum, Operation::Product, Operation::Min, Operation::Max,
                    Operation::Band, Operation::Bor, Operation::Bxor}) {
      for (size_t count : {size_t(0), size_t(1), static_cast<size_t>(config.world_size + 3), size_t(257)}) {
        if (!allreduceCase<int64_t>(collective, config.rank, config.world_size, count, op)) {
          return 4;
        }
      }
    }
    for (size_t count : {size_t(0), size_t(1), static_cast<size_t>(config.world_size - 1),
                         static_cast<size_t>(config.world_size + 3), size_t(257)}) {
      for (bool alias : {false, true}) {
        if (!scatterCase<float>(collective, config.rank, config.world_size, count, alias) ||
            !scatterCase<int64_t>(collective, config.rank, config.world_size, count, alias)) {
          return 5;
        }
      }
    }
    std::cout << "ALL TESTS PASS rank=" << config.rank << " world=" << config.world_size
              << " reduce_parallel=" << (reduceParallel ? "true" : "false") << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test exception: " << error.what() << "\n";
    return 6;
  }
}

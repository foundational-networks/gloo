#include "gloo/transport/peel/peel_allreduce_ring_sharded.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

namespace gloo {
namespace transport {
namespace peel {
namespace {

using Clock = std::chrono::steady_clock;

struct AlignedBuffer {
  std::vector<std::max_align_t> words;
  explicit AlignedBuffer(size_t bytes)
      : words(bytes / sizeof(std::max_align_t) +
              (bytes % sizeof(std::max_align_t) != 0)) {}
  void* data() { return words.data(); }
};

size_t shardSize(size_t elements, int world) {
  return elements / static_cast<size_t>(world) +
      (elements % static_cast<size_t>(world) != 0);
}

size_t validCount(size_t elements, size_t shard, int owner) {
  const size_t start = static_cast<size_t>(owner) * shard;
  return start < elements ? std::min(shard, elements - start) : 0;
}

} // namespace

PeelAllreduceRingShardedGeneric::PeelAllreduceRingShardedGeneric(
    int rank, std::vector<PeelRingHop> hops)
    : rank_(rank),
      worldSize_(std::max(1, static_cast<int>(hops.size()))),
      hops_(std::move(hops)) {}

bool PeelAllreduceRingShardedGeneric::isReady() const {
  if (rank_ < 0 || rank_ >= worldSize_) {
    return false;
  }
  if (worldSize_ == 1) {
    return true;
  }
  const auto& outgoing = hops_[rank_];
  const int predecessor = (rank_ + worldSize_ - 1) % worldSize_;
  const auto& incoming = hops_[predecessor];
  return outgoing.sender == rank_ &&
      outgoing.receiver == (rank_ + 1) % worldSize_ &&
      incoming.sender == predecessor && incoming.receiver == rank_ &&
      outgoing.transport != nullptr && incoming.transport != nullptr &&
      outgoing.transport != incoming.transport &&
      outgoing.transport->isReady() && incoming.transport->isReady();
}

size_t PeelAllreduceRingShardedGeneric::localShardElements(size_t elements) const {
  if (rank_ < 0 || rank_ >= worldSize_) {
    return 0;
  }
  return validCount(elements, shardSize(elements, worldSize_), rank_);
}

bool PeelAllreduceRingShardedGeneric::checkArguments(
    const void* input, size_t elements, size_t elementSize,
    const ReduceFunction& reduce) const {
  if (!isReady() || elementSize == 0 ||
      elements > std::numeric_limits<size_t>::max() / elementSize ||
      (elements != 0 && (input == nullptr || !reduce))) {
    std::cerr << "peel_allreduce_ring_sharded: invalid argument or hop\n";
    return false;
  }
  const size_t shard = shardSize(elements, worldSize_);
  return shard <= std::numeric_limits<size_t>::max() /
      static_cast<size_t>(worldSize_);
}

bool PeelAllreduceRingShardedGeneric::exchange(
    void* outgoing, void* incoming, size_t bytes) {
  const auto& sendHop = hops_[rank_];
  const auto& receiveHop = hops_[(rank_ + worldSize_ - 1) % worldSize_];
  receiveHop.transport->submitWork(receiveHop.sender, incoming, bytes);
  sendHop.transport->submitWork(sendHop.sender, outgoing, bytes);
  // Always drain both workers, including on a failed receive. Otherwise the
  // outgoing buffer could be destroyed while a send still accesses it.
  const bool received = receiveHop.transport->waitResult();
  const bool sent = sendHop.transport->waitResult();
  return received && sent;
}

bool PeelAllreduceRingShardedGeneric::reduceScatter(
    const void* input, void* localOutput, size_t elements, size_t elementSize,
    const ReduceFunction& reduce) {
  stats_ = Stats{};
  if (!checkArguments(input, elements, elementSize, reduce)) {
    return false;
  }
  stats_.localShardElements = localShardElements(elements);
  if (stats_.localShardElements != 0 && localOutput == nullptr) {
    return false;
  }
  if (elements == 0) {
    return true;
  }
  const size_t bytes = elements * elementSize;
  const size_t outputBytes = stats_.localShardElements * elementSize;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(input);
  const uintptr_t output = reinterpret_cast<uintptr_t>(localOutput);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes ||
      output > std::numeric_limits<uintptr_t>::max() - outputBytes) {
    return false;
  }
  const size_t shard = shardSize(elements, worldSize_);
  const size_t start = std::min(elements, static_cast<size_t>(rank_) * shard);
  const uintptr_t ownStart = begin + start * elementSize;
  if (outputBytes != 0 && output < begin + bytes &&
      output + outputBytes > begin && output != ownStart) {
    std::cerr << "peel_allreduce_ring_sharded: output overlaps another shard\n";
    return false;
  }
  const auto beginTime = Clock::now();
  const bool ok = reduceScatterImpl(input, localOutput, elements, elementSize, reduce);
  stats_.reduceScatterSeconds =
      std::chrono::duration<double>(Clock::now() - beginTime).count();
  return ok;
}

bool PeelAllreduceRingShardedGeneric::reduceScatterImpl(
    const void* input, void* localOutput, size_t elements, size_t elementSize,
    const ReduceFunction& reduce) {
  if (worldSize_ == 1) {
    if (input != localOutput) {
      std::memcpy(localOutput, input, elements * elementSize);
    }
    return true;
  }
  const size_t shard = shardSize(elements, worldSize_);
  const size_t bytes = shard * elementSize;
  const auto* inputBytes = static_cast<const uint8_t*>(input);
  AlignedBuffer outgoing(bytes);
  AlignedBuffer incoming(bytes);
  const int firstShard = (rank_ + worldSize_ - 1) % worldSize_;
  const size_t firstValid = validCount(elements, shard, firstShard);
  std::memset(outgoing.data(), 0, bytes);
  if (firstValid != 0) {
    std::memcpy(outgoing.data(),
                inputBytes + static_cast<size_t>(firstShard) * shard * elementSize,
                firstValid * elementSize);
  }
  for (int round = 0; round < worldSize_ - 1; ++round) {
    // Send (rank-round-1)%N and receive (rank-round-2)%N. After N-1
    // rounds the received shard is rank, with exactly one contribution per
    // source. Ring reduction order is cyclic, so floating roundoff can differ
    // from the SDN global-rank order while preserving mathematical semantics.
    const int receivedShard = (rank_ + 2 * worldSize_ - round - 2) % worldSize_;
    ++stats_.reduceScatterRounds;
    stats_.reduceScatterSendBytes += bytes;
    if (!exchange(outgoing.data(), incoming.data(), bytes)) {
      return false;
    }
    const size_t valid = validCount(elements, shard, receivedShard);
    // The outgoing send is complete before its storage is reused. Use the
    // established output==input1 callback contract, not input2 aliasing.
    std::memset(outgoing.data(), 0, bytes);
    if (valid != 0) {
      std::memcpy(outgoing.data(),
                  inputBytes + static_cast<size_t>(receivedShard) * shard * elementSize,
                  valid * elementSize);
      reduce(outgoing.data(), outgoing.data(), incoming.data(), valid);
    }
  }
  if (stats_.localShardElements != 0) {
    std::memcpy(localOutput, outgoing.data(), stats_.localShardElements * elementSize);
  }
  return true;
}

bool PeelAllreduceRingShardedGeneric::gatherShards(
    void* data, size_t elements, size_t elementSize) {
  if (worldSize_ == 1 || elements == 0) {
    return true;
  }
  const size_t shard = shardSize(elements, worldSize_);
  const size_t bytes = shard * elementSize;
  auto* dataBytes = static_cast<uint8_t*>(data);
  AlignedBuffer outgoing(bytes);
  AlignedBuffer incoming(bytes);
  std::memset(outgoing.data(), 0, bytes);
  const size_t ownValid = localShardElements(elements);
  if (ownValid != 0) {
    std::memcpy(outgoing.data(), dataBytes + static_cast<size_t>(rank_) * shard * elementSize,
                ownValid * elementSize);
  }
  for (int round = 0; round < worldSize_ - 1; ++round) {
    // Send (rank-round)%N, receive (rank-round-1)%N, then forward the
    // just-received complete shard without reducing it.
    const int receivedShard = (rank_ + worldSize_ - round - 1) % worldSize_;
    ++stats_.allgatherRounds;
    stats_.allgatherSendBytes += bytes;
    if (!exchange(outgoing.data(), incoming.data(), bytes)) {
      return false;
    }
    const size_t valid = validCount(elements, shard, receivedShard);
    if (valid != 0) {
      std::memcpy(dataBytes + static_cast<size_t>(receivedShard) * shard * elementSize,
                  incoming.data(), valid * elementSize);
    }
    outgoing.words.swap(incoming.words);
  }
  return true;
}

bool PeelAllreduceRingShardedGeneric::run(
    void* data, size_t elements, size_t elementSize, const ReduceFunction& reduce) {
  stats_ = Stats{};
  if (!checkArguments(data, elements, elementSize, reduce)) {
    return false;
  }
  if (elements == 0) {
    return true;
  }
  const size_t shard = shardSize(elements, worldSize_);
  const size_t start = std::min(elements, static_cast<size_t>(rank_) * shard);
  auto* localOutput = static_cast<uint8_t*>(data) + start * elementSize;
  if (!reduceScatter(data, localOutput, elements, elementSize, reduce)) {
    return false;
  }
  const auto beginTime = Clock::now();
  const bool ok = gatherShards(data, elements, elementSize);
  stats_.allgatherSeconds =
      std::chrono::duration<double>(Clock::now() - beginTime).count();
  return ok;
}

} // namespace peel
} // namespace transport
} // namespace gloo

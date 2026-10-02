// Internal bridge used by the ordinary Gloo CPU collective entry points.
// PEEL sockets are cached for one Gloo context per collective kind. Callers
// must use one ProcessGroupGloo worker and one device (see the Python helper).
#pragma once

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>

#include "gloo/common/logging.h"
#include "gloo/config.h"
#include "gloo/context.h"

namespace gloo {
namespace peel_bridge {

enum class Collective { Broadcast, Allgather, Allreduce };
enum class Algorithm {
  Default,
  GlooRing,
  Native,
  NativeReno,
  Ring,
  RingReno,
  BinomialStopAndWait,
};

inline std::string env(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

inline long integerEnv(const char* name, long fallback, long lo, long hi) {
  const auto value = env(name);
  if (value.empty()) return fallback;
  errno = 0;
  char* end = nullptr;
  const long result = std::strtol(value.c_str(), &end, 10);
  GLOO_ENFORCE(errno == 0 && end != value.c_str() && *end == '\0' &&
                   result >= lo && result <= hi,
               name, " must be an integer in [", lo, ", ", hi,
               "]; got ", value);
  return result;
}

inline double realEnv(const char* name, double fallback, double lo, double hi) {
  const auto value = env(name);
  if (value.empty()) return fallback;
  errno = 0;
  char* end = nullptr;
  const double result = std::strtod(value.c_str(), &end);
  GLOO_ENFORCE(errno == 0 && end != value.c_str() && *end == '\0' &&
                   std::isfinite(result) && result >= lo && result <= hi,
               name, " must be a number in [", lo, ", ", hi,
               "]; got ", value);
  return result;
}

inline bool boolEnv(const char* name, bool fallback) {
  const auto value = env(name);
  if (value.empty()) return fallback;
  if (value == "1" || value == "true" || value == "TRUE" ||
      value == "yes" || value == "YES") return true;
  if (value == "0" || value == "false" || value == "FALSE" ||
      value == "no" || value == "NO") return false;
  GLOO_ENFORCE(false, "Invalid boolean for ", name, ": ", value);
  return fallback;
}

inline const char* selector(Collective kind) {
  switch (kind) {
    case Collective::Broadcast: return "GLOO_BROADCAST_ALGORITHM";
    case Collective::Allgather: return "GLOO_ALLGATHER_ALGORITHM";
    case Collective::Allreduce: return "GLOO_ALLREDUCE_ALGORITHM";
  }
  return "";
}

inline const char* operationName(Collective kind) {
  switch (kind) {
    case Collective::Broadcast: return "broadcast";
    case Collective::Allgather: return "allgather";
    case Collective::Allreduce: return "allreduce";
  }
  return "";
}

inline Algorithm selectedAlgorithm(Collective kind) {
  const auto value = env(selector(kind));
  const std::string op = operationName(kind);
  if (value.empty() || value == "default" || value == op)
    return Algorithm::Default;
  if (kind == Collective::Broadcast &&
      (value == "ring" || value == "broadcast_ring"))
    return Algorithm::GlooRing;
  // Keep the old broadcast/allreduce "peel" aliases mapped to stop-and-wait
  // rings. Explicit names are recommended for reproducible experiments.
  if (value == "peel" || value == "peel_" + op + "_ring")
    return Algorithm::Ring;
  if (value == "peel_" + op + "_ring_reno")
    return Algorithm::RingReno;
  if (kind != Collective::Allreduce) {
    if (value == "peel_" + op) return Algorithm::Native;
    if (value == "peel_" + op + "_reno") return Algorithm::NativeReno;
  }
  if (kind == Collective::Broadcast &&
      value == "peel_broadcast_stop_and_wait")
    return Algorithm::BinomialStopAndWait;
  GLOO_ENFORCE(false, "Unsupported ", selector(kind), ": ", value);
  return Algorithm::Default;
}

inline bool isPeel(Algorithm algorithm) {
  return algorithm != Algorithm::Default && algorithm != Algorithm::GlooRing;
}

inline bool needsTopology(Algorithm algorithm) {
  return algorithm == Algorithm::Native || algorithm == Algorithm::NativeReno;
}

inline bool usesReno(Algorithm algorithm) {
  return algorithm == Algorithm::NativeReno || algorithm == Algorithm::RingReno;
}

inline const char* algorithmName(Collective kind, Algorithm algorithm) {
  if (kind == Collective::Broadcast) {
    switch (algorithm) {
      case Algorithm::Native: return "peel_broadcast";
      case Algorithm::NativeReno: return "peel_broadcast_reno";
      case Algorithm::Ring: return "peel_broadcast_ring";
      case Algorithm::RingReno: return "peel_broadcast_ring_reno";
      case Algorithm::BinomialStopAndWait: return "peel_broadcast_stop_and_wait";
      default: return "default";
    }
  }
  if (kind == Collective::Allgather) {
    switch (algorithm) {
      case Algorithm::Native: return "peel_allgather";
      case Algorithm::NativeReno: return "peel_allgather_reno";
      case Algorithm::Ring: return "peel_allgather_ring";
      case Algorithm::RingReno: return "peel_allgather_ring_reno";
      default: return "default";
    }
  }
  return usesReno(algorithm) ? "peel_allreduce_ring_reno" : "peel_allreduce_ring";
}

inline const char* portVariable(Collective kind) {
  switch (kind) {
    case Collective::Broadcast: return "GLOO_PEEL_BROADCAST_BASE_PORT";
    case Collective::Allgather: return "GLOO_PEEL_ALLGATHER_BASE_PORT";
    case Collective::Allreduce: return "GLOO_PEEL_ALLREDUCE_BASE_PORT";
  }
  return "";
}

inline long basePort(Collective kind) {
  const long fallback = 50000 + 1000 * static_cast<int>(kind);
  return env(portVariable(kind)).empty()
      ? integerEnv("GLOO_PEEL_BASE_PORT", fallback, 1, 65535)
      : integerEnv(portVariable(kind), fallback, 1, 65535);
}

// Reserve every possible root's ports: native trees already add root*W^2
// inside PeelTree; the binomial bridge adds that offset itself. Rings use W^2.
inline uint64_t portSpan(Algorithm algorithm, int worldSize) {
  const uint64_t w = static_cast<uint64_t>(worldSize);
  return (needsTopology(algorithm) ||
          algorithm == Algorithm::BinomialStopAndWait) ? w * w * w : w * w;
}

inline void validatePortRanges(int worldSize) {
  GLOO_ENFORCE(worldSize >= 1 && worldSize <= 256,
               "PEEL's packet rank field supports world sizes 1..256");
  for (int i = 0; i < 3; ++i) {
    const auto kind = static_cast<Collective>(i);
    const auto algorithm = selectedAlgorithm(kind);
    if (!isPeel(algorithm)) continue;
    const uint64_t first = basePort(kind);
    const uint64_t last = first + portSpan(algorithm, worldSize) - 1;
    GLOO_ENFORCE(last <= 65535, portVariable(kind),
                 " reserves ports through ", last, "; reduce the base port");
    for (int j = 0; j < i; ++j) {
      const auto other = static_cast<Collective>(j);
      const auto otherAlgorithm = selectedAlgorithm(other);
      if (!isPeel(otherAlgorithm)) continue;
      const uint64_t otherFirst = basePort(other);
      const uint64_t otherLast = otherFirst + portSpan(otherAlgorithm, worldSize) - 1;
      GLOO_ENFORCE(last < otherFirst || otherLast < first,
                   "Overlapping PEEL port ranges: ", portVariable(other),
                   " and ", portVariable(kind),
                   ". Set distinct collective-specific base ports.");
    }
  }
}

} // namespace peel_bridge
} // namespace gloo

#if GLOO_HAVE_TRANSPORT_PEEL

#include <arpa/inet.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include "gloo/transport/peel/peel_allgather.h"
#include "gloo/transport/peel/peel_allreduce_ring.h"
#include "gloo/transport/peel/peel_context.h"
#include "gloo/transport/unbound_buffer.h"
#include "gloo/types.h"

namespace gloo {
namespace peel_bridge {

inline std::string interfaceName() {
  auto iface = env("GLOO_PEEL_IFACE");
  if (iface.empty()) {
    iface = env("GLOO_SOCKET_IFNAME");
    iface = iface.substr(0, iface.find(','));
  }
  GLOO_ENFORCE(!iface.empty() && iface.size() < IFNAMSIZ,
               "Set GLOO_PEEL_IFACE to one IPv4 interface");
  return iface;
}

inline uint32_t interfaceIp(const std::string& iface) {
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  GLOO_ENFORCE(fd >= 0, "Cannot open interface query socket");
  ifreq request{};
  std::copy(iface.begin(), iface.end(), request.ifr_name);
  const int result = ::ioctl(fd, SIOCGIFADDR, &request);
  ::close(fd);
  GLOO_ENFORCE(result == 0, "Cannot obtain IPv4 address for ", iface);
  return reinterpret_cast<const sockaddr_in*>(&request.ifr_addr)->sin_addr.s_addr;
}

inline uint64_t hashBytes(const std::string& data) {
  uint64_t hash = 14695981039346656037ULL;
  for (unsigned char value : data) {
    hash ^= value;
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline transport::peel::PeelContextConfig makeConfig(
    const std::shared_ptr<Context>& context,
    Collective kind,
    Algorithm algorithm,
    std::chrono::milliseconds timeout) {
  validatePortRanges(context->size);
  transport::peel::PeelContextConfig config;
  config.rank = context->rank;
  config.world_size = context->size;
  config.iface_name = interfaceName();
  config.base_port = static_cast<uint16_t>(basePort(kind));
  const auto group = env("GLOO_PEEL_MCAST_GROUP");
  if (!group.empty()) config.mcast_group = group;
  in_addr address{};
  GLOO_ENFORCE(::inet_pton(AF_INET, config.mcast_group.c_str(), &address) == 1 &&
                   IN_MULTICAST(ntohl(address.s_addr)),
               "GLOO_PEEL_MCAST_GROUP must be an IPv4 multicast address");
  config.ttl = static_cast<int>(integerEnv("GLOO_PEEL_TTL", 64, 1, 255));
  config.dscp = static_cast<uint8_t>(integerEnv("GLOO_PEEL_DSCP", 7, 0, 63));
  config.rto_ms = static_cast<int>(integerEnv("GLOO_PEEL_RTO_MS", 500, 1, 30000));
  const long defaultTimeout = timeout.count() > 0
      ? static_cast<long>(std::min<int64_t>(timeout.count(), INT_MAX)) : 300000;
  config.timeout_ms = static_cast<int>(integerEnv(
      "GLOO_PEEL_TIMEOUT_MS", defaultTimeout, 1, INT_MAX));
  config.rcvbuf = static_cast<int>(integerEnv(
      "GLOO_PEEL_RCVBUF", 32 * 1024 * 1024, 1, INT_MAX));
  config.max_chunk_size = static_cast<size_t>(integerEnv(
      "GLOO_PEEL_MAX_PAYLOAD", 0, 0,
      65535 - 20 - 8 - transport::peel::PEEL_HEADER_SIZE));
  config.reno_dupack_pct = static_cast<float>(realEnv(
      "GLOO_PEEL_RENO_DUPACK_PCT", 50.0, 0.000001, 100.0));
  config.reno_tagg_ms = static_cast<int>(integerEnv(
      "GLOO_PEEL_RENO_TAGG_MS", 100, 1, INT_MAX));
  config.reno_ooo_buffer_segments = static_cast<uint32_t>(integerEnv(
      "GLOO_PEEL_RENO_OOO_BUF", 64, 1, 65535));
  config.reno_rto_reset_on_ack = boolEnv("GLOO_PEEL_RENO_RTO_RESET_ON_ACK", true);
  if (needsTopology(algorithm)) {
    config.topology_file = env("GLOO_PEEL_TOPOLOGY_FILE");
    GLOO_ENFORCE(!config.topology_file.empty(),
                 algorithmName(kind, algorithm),
                 " requires GLOO_PEEL_TOPOLOGY_FILE; flat fallback is disabled");
  }
  return config;
}

// Small setup messages travel over the existing TCP Gloo connections. They
// never call gloo::allgather(), avoiding recursive PEEL dispatch. Prefixes
// 0x70..0x72 are private to this bridge; ordinary Gloo uses 0x01..0x08.
template <typename T>
inline std::vector<T> exchangeSetup(
    const std::shared_ptr<Context>& context, Collective kind, uint32_t tag,
    const T& local, std::chrono::milliseconds timeout) {
  std::vector<T> records(context->size);
  records[context->rank] = local;
  auto buffer = context->createUnboundBuffer(records.data(), records.size() * sizeof(T));
  const auto slot = Slot::build(0x70 + static_cast<uint8_t>(kind), tag);
  const int next = (context->rank + 1) % context->size;
  const int prev = (context->rank + context->size - 1) % context->size;
  for (int step = 0; step < context->size - 1; ++step) {
    const int src = (context->rank - step + context->size) % context->size;
    const int dst = (src + context->size - 1) % context->size;
    buffer->recv(prev, slot, dst * sizeof(T), sizeof(T));
    buffer->send(next, slot, src * sizeof(T), sizeof(T));
    GLOO_ENFORCE(buffer->waitRecv(timeout), "PEEL setup receive aborted");
    GLOO_ENFORCE(buffer->waitSend(timeout), "PEEL setup send aborted");
  }
  return records;
}

inline void setupBarrier(const std::shared_ptr<Context>& context,
                         Collective kind, uint32_t tag,
                         std::chrono::milliseconds timeout) {
  auto buffer = context->createUnboundBuffer(nullptr, 0);
  const auto slot = Slot::build(0x70 + static_cast<uint8_t>(kind), tag) + uint8_t{1};
  for (int d = 1; d < context->size; d <<= 1) {
    buffer->recv((context->rank + context->size - d) % context->size, slot);
    buffer->send((context->rank + d) % context->size, slot);
    GLOO_ENFORCE(buffer->waitRecv(timeout), "PEEL setup barrier aborted");
    GLOO_ENFORCE(buffer->waitSend(timeout), "PEEL setup barrier aborted");
  }
}

class Runtime {
 public:
  static Runtime& instance(Collective kind) {
    static Runtime broadcast(Collective::Broadcast);
    static Runtime allgather(Collective::Allgather);
    static Runtime allreduce(Collective::Allreduce);
    switch (kind) {
      case Collective::Broadcast: return broadcast;
      case Collective::Allgather: return allgather;
      case Collective::Allreduce: return allreduce;
    }
    return broadcast;
  }

  void broadcast(const std::shared_ptr<Context>& context, Algorithm algorithm,
                 int root, void* data, size_t bytes, uint32_t tag,
                 std::chrono::milliseconds timeout) {
    auto lock = operationLock();
    initialize(context, algorithm, tag, timeout);
    auto& peel = ensureContext(context, root, tag, timeout);
    trace(tag, bytes);
    bool ok;
    if (needsTopology(algorithm)) ok = peel.broadcast(root, data, bytes);
    else if (algorithm == Algorithm::BinomialStopAndWait)
      ok = peel.broadcastStopAndWait(root, data, bytes);
    else ok = peel.broadcastRing(root, data, bytes);
    GLOO_ENFORCE(ok, algorithmName(kind_, algorithm), " failed");
  }

  void allgather(const std::shared_ptr<Context>& context, Algorithm algorithm,
                 const std::vector<void*>& buffers, size_t bytes, uint32_t tag,
                 std::chrono::milliseconds timeout) {
    auto lock = operationLock();
    initialize(context, algorithm, tag, timeout);
    bool ok;
    if (needsTopology(algorithm)) {
      std::vector<transport::peel::PeelContext*> roots;
      for (int root = 0; root < context->size; ++root)
        roots.push_back(&ensureContext(context, root, tag, timeout));
      transport::peel::PeelAllgather operation(
          roots, parallelAllgather_ ? transport::peel::PeelAllgatherMode::Parallel
                                   : transport::peel::PeelAllgatherMode::Sequential);
      trace(tag, bytes);
      ok = operation.run(buffers, bytes);
    } else {
      auto& peel = ensureContext(context, 0, tag, timeout);
      trace(tag, bytes);
      ok = peel.allgatherRing(buffers, bytes);
    }
    GLOO_ENFORCE(ok, algorithmName(kind_, algorithm), " failed");
  }

  void allreduce(const std::shared_ptr<Context>& context, Algorithm algorithm,
                 void* data, size_t elements, size_t elementSize,
                 transport::peel::PeelAllreduceRingGeneric::ReduceFunction reduce,
                 uint32_t tag, std::chrono::milliseconds timeout) {
    auto lock = operationLock();
    initialize(context, algorithm, tag, timeout);
    auto& peel = ensureContext(context, 0, tag, timeout);
    trace(tag, elements * elementSize);
    transport::peel::PeelAllreduceRingGeneric operation(
        context->rank, peel.ringHops(), {data}, elements, elementSize, std::move(reduce));
    GLOO_ENFORCE(operation.run(), algorithmName(kind_, algorithm), " failed");
  }

 private:
  explicit Runtime(Collective kind) : kind_(kind) {}

  std::unique_lock<std::mutex> operationLock() {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    GLOO_ENFORCE(lock.owns_lock(),
                 "Concurrent PEEL operations of the same kind are unsupported. "
                 "Use pytorch_peel_backend.init_process_group (one Gloo worker).");
    return lock;
  }

  void initialize(const std::shared_ptr<Context>& context, Algorithm algorithm,
                  uint32_t tag, std::chrono::milliseconds timeout) {
    if (initialized_) {
      GLOO_ENFORCE(boundContext_.lock() == context,
                   "PEEL supports one Gloo context per collective kind per process. "
                   "Use one device, one process group, and restart between runs.");
      GLOO_ENFORCE(algorithm_ == algorithm,
                   "Set PEEL selectors before initialization; restart to change algorithms");
      return;
    }
    config_ = makeConfig(context, kind_, algorithm, timeout);
    const auto mode = env("GLOO_PEEL_ALLGATHER_MODE");
    GLOO_ENFORCE(mode.empty() || mode == "parallel" || mode == "sequential",
                 "GLOO_PEEL_ALLGATHER_MODE must be parallel or sequential");
    parallelAllgather_ = mode == "parallel";

    std::string topology;
    if (needsTopology(algorithm)) {
      std::ifstream file(config_.topology_file, std::ios::binary);
      GLOO_ENFORCE(file.is_open(), "Cannot read GLOO_PEEL_TOPOLOGY_FILE: ",
                   config_.topology_file);
      topology.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
      GLOO_ENFORCE(!topology.empty(), "PEEL topology file is empty");
    }
    // Stable configuration fingerprint: local interface names and local file
    // paths may differ; wire settings and topology bytes must agree.
    std::string settings = std::to_string(static_cast<int>(algorithm)) + ":" +
        config_.mcast_group + ":" + std::to_string(config_.base_port) + ":" +
        std::to_string(config_.ttl) + ":" + std::to_string(config_.dscp) + ":" +
        std::to_string(config_.max_chunk_size) + ":" +
        std::to_string(config_.reno_dupack_pct) + ":" +
        std::to_string(config_.reno_tagg_ms) + ":" +
        std::to_string(config_.reno_ooo_buffer_segments) + ":" +
        std::to_string(config_.reno_rto_reset_on_ack) + ":" +
        std::to_string(parallelAllgather_) + ":" + topology;
    struct SetupRecord { uint64_t fingerprint; uint32_t ip; uint32_t reserved; };
    const SetupRecord local{hashBytes(settings), interfaceIp(config_.iface_name), 0};
    const auto records = exchangeSetup(context, kind_, tag, local, timeout);
    std::unordered_set<uint32_t> uniqueIps;
    for (int rank = 0; rank < context->size; ++rank) {
      GLOO_ENFORCE(records[rank].fingerprint == local.fingerprint,
                   "PEEL settings/topology differ at rank ", rank);
      if (needsTopology(algorithm)) {
        GLOO_ENFORCE(uniqueIps.insert(records[rank].ip).second,
                     "Native PEEL requires a distinct topology IPv4 address per rank. "
                     "Launch one rank per host/NIC address.");
      }
      char ip[INET_ADDRSTRLEN]{};
      GLOO_ENFORCE(::inet_ntop(AF_INET, &records[rank].ip, ip, sizeof(ip)),
                   "Invalid PEEL rank IPv4 address");
      config_.peer_ips[rank] = ip;
    }
    algorithm_ = algorithm;
    contexts_.resize(needsTopology(algorithm) ||
        algorithm == Algorithm::BinomialStopAndWait ? context->size : 1);
    boundContext_ = context;
    trace_ = boolEnv("GLOO_PEEL_TRACE", false);
    initialized_ = true;
  }

  transport::peel::PeelContext& ensureContext(
      const std::shared_ptr<Context>& context, int root, uint32_t tag,
      std::chrono::milliseconds timeout) {
    const bool perRoot = needsTopology(algorithm_) ||
        algorithm_ == Algorithm::BinomialStopAndWait;
    const int index = perRoot ? root : 0;
    if (contexts_[index]) return *contexts_[index];
    auto config = config_;
    config.sender_rank = root;
    if (algorithm_ == Algorithm::BinomialStopAndWait)
      config.base_port = static_cast<uint16_t>(config_.base_port +
          root * context->size * context->size);
    auto peel = std::make_unique<transport::peel::PeelContext>(config);
    bool ok;
    if (needsTopology(algorithm_)) ok = usesReno(algorithm_) ? peel->initReno() : peel->init();
    else if (algorithm_ == Algorithm::BinomialStopAndWait) ok = peel->initStopAndWait();
    else ok = usesReno(algorithm_) ? peel->initRingReno() : peel->initRing();
    GLOO_ENFORCE(ok, "PEEL initialization failed for ", algorithmName(kind_, algorithm_),
                 " root=", root);
    // Initialization handshakes and tree construction finish before data
    // transfer. This barrier runs only once per cached context/root.
    setupBarrier(context, kind_, tag, timeout);
    contexts_[index] = std::move(peel);
    return *contexts_[index];
  }

  void trace(uint32_t tag, size_t bytes) {
    if (trace_)
      std::cerr << "gloo " << algorithmName(kind_, algorithm_)
                << ": rank=" << config_.rank << " tag=" << tag
                << " bytes=" << bytes << "\n";
  }

  Collective kind_;
  Algorithm algorithm_ = Algorithm::Default;
  std::mutex mutex_;
  bool initialized_ = false;
  bool parallelAllgather_ = false;
  bool trace_ = false;
  std::weak_ptr<Context> boundContext_;
  transport::peel::PeelContextConfig config_;
  std::vector<std::unique_ptr<transport::peel::PeelContext>> contexts_;
};

} // namespace peel_bridge
} // namespace gloo

#endif // GLOO_HAVE_TRANSPORT_PEEL

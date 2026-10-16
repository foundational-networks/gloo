#pragma once

#include "gloo/transport/peel/peel_broadcast_ring.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace gloo {
namespace transport {
namespace peel {

// Bandwidth-optimal sharded ring over initialized PEEL Reno ring hops.
// The existing full-tensor PeelAllreduceRingGeneric remains unchanged.
// No SDN multicast is claimed for this explicit ring comparator.
class PeelAllreduceRingShardedGeneric {
 public:
  using ReduceFunction =
      std::function<void(void*, const void*, const void*, size_t)>;

  struct Stats {
    double reduceScatterSeconds = 0;
    double allgatherSeconds = 0;
    // Logical padded payload sent by this rank; excludes framing/ACK/retries.
    uint64_t reduceScatterSendBytes = 0;
    uint64_t allgatherSendBytes = 0;
    uint64_t reduceScatterRounds = 0;
    uint64_t allgatherRounds = 0;
    size_t localShardElements = 0;
  };

  // hops contains one global directed edge r -> (r+1)%N for every rank.
  // Only the local incoming and outgoing transports need to be nonnull.
  // An empty hops vector is the existing one-rank context convention.
  PeelAllreduceRingShardedGeneric(int rank, std::vector<PeelRingHop> hops);

  bool isReady() const;
  bool run(void* data, size_t elements, size_t elementSize,
           const ReduceFunction& reduce);

  // Input remains immutable with separate output. Output may instead alias
  // exactly this rank's shard start; it is written only after the final round.
  // ceil(elements/N) shards, truncated at elements. Padding is never reduced.
  bool reduceScatter(const void* input, void* localOutput, size_t elements,
                     size_t elementSize, const ReduceFunction& reduce);
  size_t localShardElements(size_t elements) const;
  const Stats& lastStats() const { return stats_; }

 private:
  bool checkArguments(const void* input, size_t elements, size_t elementSize,
                      const ReduceFunction& reduce) const;
  bool reduceScatterImpl(const void* input, void* localOutput, size_t elements,
                         size_t elementSize, const ReduceFunction& reduce);
  bool gatherShards(void* data, size_t elements, size_t elementSize);
  bool exchange(void* outgoing, void* incoming, size_t bytes);

  int rank_;
  int worldSize_;
  std::vector<PeelRingHop> hops_;
  Stats stats_;
};

} // namespace peel
} // namespace transport
} // namespace gloo

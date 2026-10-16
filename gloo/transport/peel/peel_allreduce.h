#pragma once

#include "gloo/transport/peel/peel_allgather.h"
#include "gloo/transport/peel/peel_context.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace gloo {
namespace transport {
namespace peel {

// Topology-routed SDN reduce-scatter followed by parallel SDN allgather.
// Each consecutive pair of shard owners receives a source's two shard tiles
// through one PeelTree broadcast. Roots outside that pair therefore multicast
// to two owners. Owners reduce independent received contributions in global
// rank order. The singleton tail for odd world sizes uses the same topology
// route with one receiver. All data transfers use userspace PEEL Reno.
//
// The caller serializes operations and performs its usual collective setup
// barrier after initReno(). No TCP payload or ring fallback is used here.
class PeelAllreduce {
 public:
  using ReduceFunction =
      std::function<void(void*, const void*, const void*, size_t)>;

  // Broadcast input bytes are logical payload counts before subtree fanout,
  // framing, acknowledgments, retransmissions, or Ethernet replication.
  // Broadcast counts include this rank's sends and receives.
  struct Stats {
    double reduceScatterSeconds = 0;
    double allgatherSeconds = 0;
    uint64_t reduceScatterSendBytes = 0;
    uint64_t allgatherSendBytes = 0;
    uint64_t reduceScatterBroadcasts = 0;
    uint64_t allgatherBroadcasts = 0;
    size_t localShardElements = 0;
  };

  explicit PeelAllreduce(
      const PeelContextConfig& config,
      size_t tileBytes = 256 * 1024);

  bool initReno();
  bool isReady() const;

  // Opt-in owner-pair concurrency for standalone experiments. All ranks use
  // the same setting, and the caller changes it only between operations.
  // Sequential groups remain the default; initialization is unchanged.
  void setReduceGroupsParallel(bool enabled) { reduceGroupsParallel_ = enabled; }

  // In-place allreduce of one contiguous input. The reduction callback must
  // support output == input1, as required by existing Gloo allreduce paths.
  // Padding is transferred but never passed to the callback or copied out.
  bool run(
      void* data,
      size_t elements,
      size_t elementSize,
      const ReduceFunction& reduce);

  // ceil(elements/worldSize) shards, with the last shards truncated as needed.
  // localOutput has localShardElements(elements) entries. It may be disjoint
  // from input, or exactly the rank's shard start inside input; other overlap
  // is rejected. The rank's unowned input shards remain untouched.
  bool reduceScatter(
      const void* input,
      void* localOutput,
      size_t elements,
      size_t elementSize,
      const ReduceFunction& reduce);

  size_t localShardElements(size_t elements) const;
  const Stats& lastStats() const { return stats_; }

  // RS source r owns [base+r*N^2, base+(r+1)*N^2). A subgroup's root is
  // remapped to zero and group blocks consume at most N^2 ports per source.
  // Full-world allgather starts at base+N^3 and occupies another N^3 ports.
  static uint64_t portSpan(int worldSize);

 private:
  struct OwnerGroup {
    int first = 0;
    int count = 0;
    uint64_t portOffset = 0;
    uint64_t portBlock = 0;
    std::vector<std::unique_ptr<PeelContext>> sources;
  };

  bool checkArguments(
      const void* input,
      size_t elements,
      size_t elementSize,
      const ReduceFunction& reduce) const;
  bool reduceScatterImpl(
      const void* input,
      void* localOutput,
      size_t elements,
      size_t elementSize,
      const ReduceFunction& reduce);
  // Each group owns its scratch and counters. Only this rank's owner group
  // writes localOutput; other groups read disjoint input shard ranges.
  bool reduceScatterGroup(
      OwnerGroup& group,
      const void* input,
      void* localOutput,
      size_t elements,
      size_t elementSize,
      const ReduceFunction& reduce,
      Stats& groupStats);
  bool gatherShards(void* data, size_t elements, size_t elementSize);

  PeelContextConfig config_;
  size_t tileBytes_;
  bool ready_ = false;
  bool reduceGroupsParallel_ = false;
  std::vector<OwnerGroup> groups_;
  std::vector<std::unique_ptr<PeelContext>> gatherContexts_;
  std::unique_ptr<PeelAllgather> gather_;
  Stats stats_;
};

} // namespace peel
} // namespace transport
} // namespace gloo

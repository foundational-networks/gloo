#include "gloo/transport/peel/peel_allreduce.h"
#include "gloo/transport/peel/peel_tree.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>

namespace gloo {
namespace transport {
namespace peel {
namespace {

using Clock = std::chrono::steady_clock;

// Received callback operands have ordinary scalar alignment even though the
// transport API takes untyped byte buffers.
struct AlignedBuffer {
  std::vector<std::max_align_t> words;
  void resize(size_t bytes) {
    words.resize(bytes / sizeof(std::max_align_t) +
                 (bytes % sizeof(std::max_align_t) != 0));
  }
  void* data() { return words.data(); }
  const void* data() const { return words.data(); }
};

size_t shardSize(size_t elements, int world) {
  return elements / static_cast<size_t>(world) +
      (elements % static_cast<size_t>(world) != 0);
}

size_t validCount(size_t elements, size_t shard, int owner,
                  size_t tileOffset, size_t tileCount) {
  const size_t start = static_cast<size_t>(owner) * shard;
  if (start >= elements || tileOffset >= elements - start) {
    return 0;
  }
  return std::min(tileCount, elements - start - tileOffset);
}

bool validateTree(const PeelContextConfig& config, uint64_t reservedPorts) {
  PeelTreeConfig treeConfig;
  treeConfig.rank = config.rank;
  treeConfig.world_size = config.world_size;
  treeConfig.sender_rank = config.sender_rank;
  treeConfig.peer_ips = config.peer_ips;
  treeConfig.base_port = config.base_port;
  treeConfig.topology_file = config.topology_file;
  PeelTree tree(treeConfig);
  if (!tree.loadTopology(config.topology_file) || !tree.build()) {
    return false;
  }
  std::vector<int> coverage(config.world_size, 0);
  for (const auto& subtree : tree.subtrees()) {
    const uint64_t end = static_cast<uint64_t>(subtree.base_port) +
        config.world_size;
    if (subtree.base_port < config.base_port ||
        end > static_cast<uint64_t>(config.base_port) + reservedPorts) {
      std::cerr << "peel_allreduce: subtree exceeds reserved port block\n";
      return false;
    }
    for (int rank : subtree.receiver_ranks) {
      if (rank < 0 || rank >= config.world_size) {
        return false;
      }
      if (rank != config.sender_rank) {
        ++coverage[rank];
      }
    }
  }
  // Reject unsupported direct-host links and incomplete/duplicate routes;
  // never let an omitted receiver silently bypass the SDN data transfer.
  for (int rank = 0; rank < config.world_size; ++rank) {
    if (rank != config.sender_rank && coverage[rank] != 1) {
      std::cerr << "peel_allreduce: topology does not route each receiver "
                   "exactly once\n";
      return false;
    }
  }
  return !tree.subtrees().empty();
}

} // namespace

PeelAllreduce::PeelAllreduce(const PeelContextConfig& config, size_t tileBytes)
    : config_(config), tileBytes_(tileBytes) {}

uint64_t PeelAllreduce::portSpan(int worldSize) {
  if (worldSize < 1) {
    return 0;
  }
  const uint64_t world = static_cast<uint64_t>(worldSize);
  // Any larger world cannot fit even the smallest valid UDP base. Avoid
  // arithmetic overflow before the caller validates the complete range.
  if (world > 32) {
    return std::numeric_limits<uint64_t>::max();
  }
  return 2 * world * world * world;
}

bool PeelAllreduce::initReno() {
  if (ready_) {
    return true;
  }
  const int world = config_.world_size;
  if (world < 1 || config_.rank < 0 || config_.rank >= world ||
      tileBytes_ == 0 || tileBytes_ > std::numeric_limits<size_t>::max() / 2) {
    return false;
  }
  if (world == 1) {
    ready_ = true;
    return true;
  }
  const uint64_t span = portSpan(world);
  if (span > 65536 ||
      static_cast<uint64_t>(config_.base_port) + span > 65536 ||
      config_.topology_file.empty()) {
    std::cerr << "peel_allreduce: invalid ports or missing SDN topology\n";
    return false;
  }
  for (int rank = 0; rank < world; ++rank) {
    const auto it = config_.peer_ips.find(rank);
    if (it == config_.peer_ips.end() || it->second.empty()) {
      std::cerr << "peel_allreduce: missing global peer IP\n";
      return false;
    }
  }
  groups_.clear();
  gatherContexts_.clear();
  gather_.reset();
  const uint64_t world2 = static_cast<uint64_t>(world) * world;
  const uint64_t world3 = world2 * world;
  uint64_t groupOffset = 0;
  for (int first = 0; first < world; first += 2) {
    OwnerGroup group;
    group.first = first;
    group.count = std::min(2, world - first);
    const int maxMembers = std::min(world, group.count + 1);
    group.portOffset = groupOffset;
    group.portBlock = static_cast<uint64_t>(maxMembers) * (maxMembers - 1);
    groupOffset += group.portBlock;
    if (groupOffset > world2) {
      return false;
    }
    group.sources.resize(world);
    for (int root = 0; root < world; ++root) {
      std::vector<int> members{root};
      for (int owner = first; owner < first + group.count; ++owner) {
        if (owner != root) {
          members.push_back(owner);
        }
      }
      const auto local = std::find(members.begin(), members.end(), config_.rank);
      if (local == members.end() || members.size() == 1) {
        continue;
      }
      PeelContextConfig subgroup = config_;
      subgroup.rank = static_cast<int>(local - members.begin());
      subgroup.world_size = static_cast<int>(members.size());
      subgroup.sender_rank = 0;
      subgroup.base_port = static_cast<uint16_t>(
          config_.base_port + root * world2 + group.portOffset);
      subgroup.peer_ips.clear();
      for (size_t rank = 0; rank < members.size(); ++rank) {
        subgroup.peer_ips[static_cast<int>(rank)] =
            config_.peer_ips.at(members[rank]);
      }
      if (!validateTree(subgroup, group.portBlock)) {
        return false;
      }
      auto context = std::make_unique<PeelContext>(subgroup);
      if (!context->initReno()) {
        return false;
      }
      group.sources[root] = std::move(context);
    }
    groups_.push_back(std::move(group));
  }
  std::vector<PeelContext*> rawGather;
  for (int root = 0; root < world; ++root) {
    PeelContextConfig gatherConfig = config_;
    gatherConfig.sender_rank = root;
    gatherConfig.base_port = static_cast<uint16_t>(config_.base_port + world3);
    if (!validateTree(gatherConfig, world3)) {
      return false;
    }
    auto context = std::make_unique<PeelContext>(gatherConfig);
    if (!context->initReno()) {
      return false;
    }
    rawGather.push_back(context.get());
    gatherContexts_.push_back(std::move(context));
  }
  gather_ = std::make_unique<PeelAllgather>(
      std::move(rawGather), PeelAllgatherMode::Parallel);
  ready_ = true;
  return true;
}

bool PeelAllreduce::isReady() const {
  return ready_;
}

size_t PeelAllreduce::localShardElements(size_t elements) const {
  if (config_.world_size < 1 || config_.rank < 0 ||
      config_.rank >= config_.world_size) {
    return 0;
  }
  const size_t shard = shardSize(elements, config_.world_size);
  return validCount(elements, shard, config_.rank, 0, shard);
}

bool PeelAllreduce::checkArguments(
    const void* input, size_t elements, size_t elementSize,
    const ReduceFunction& reduce) const {
  if (!ready_ || elementSize == 0 || elementSize > tileBytes_ ||
      elements > std::numeric_limits<size_t>::max() / elementSize ||
      (elements != 0 && (input == nullptr || !reduce))) {
    std::cerr << "peel_allreduce: invalid input or uninitialized context\n";
    return false;
  }
  // The last padded shard start is at most elements+worldSize-1. Reject
  // pathological size_t inputs before rank*shard pointer calculations.
  const size_t shard = shardSize(elements, config_.world_size);
  if (shard > std::numeric_limits<size_t>::max() /
          static_cast<size_t>(config_.world_size)) {
    return false;
  }
  return true;
}

bool PeelAllreduce::reduceScatter(
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
  const size_t outBytes = stats_.localShardElements * elementSize;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(input);
  const uintptr_t output = reinterpret_cast<uintptr_t>(localOutput);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes ||
      output > std::numeric_limits<uintptr_t>::max() - outBytes) {
    return false;
  }
  const size_t shard = shardSize(elements, config_.world_size);
  const size_t start = std::min(elements, static_cast<size_t>(config_.rank) * shard);
  const uintptr_t ownStart = begin + start * elementSize;
  if (outBytes != 0 && output < begin + bytes && output + outBytes > begin &&
      output != ownStart) {
    std::cerr << "peel_allreduce: output overlaps another input shard\n";
    return false;
  }
  const auto startTime = Clock::now();
  const bool ok = reduceScatterImpl(input, localOutput, elements, elementSize, reduce);
  stats_.reduceScatterSeconds =
      std::chrono::duration<double>(Clock::now() - startTime).count();
  return ok;
}

bool PeelAllreduce::reduceScatterImpl(
    const void* input, void* localOutput, size_t elements, size_t elementSize,
    const ReduceFunction& reduce) {
  if (config_.world_size == 1) {
    if (input != localOutput) {
      std::memcpy(localOutput, input, elements * elementSize);
    }
    return true;
  }
  // Group contexts have disjoint port blocks. Scratch and counters must also
  // remain disjoint until all workers join; the public operation is serialized
  // by the caller as before. In-place output belongs to exactly one group.
  std::vector<Stats> groupStats;
  std::vector<std::thread> groupThreads;
  std::atomic<bool> ok{true};
  try {
    groupStats.resize(groups_.size());
    if (reduceGroupsParallel_ && groups_.size() > 1) {
      groupThreads.reserve(groups_.size());
      const size_t shard = shardSize(elements, config_.world_size);
      for (size_t index = 0; index < groups_.size(); ++index) {
        if (validCount(elements, shard, groups_[index].first, 0, shard) == 0) {
          continue;
        }
        groupThreads.emplace_back(
            [this, input, localOutput, elements, elementSize, &reduce,
             &groupStats, &ok, index]() {
              try {
                if (!reduceScatterGroup(groups_[index], input, localOutput,
                                        elements, elementSize, reduce,
                                        groupStats[index])) {
                  ok.store(false, std::memory_order_relaxed);
                }
              } catch (const std::exception& error) {
                std::cerr << "peel_allreduce: group worker exception: "
                          << error.what() << "\n";
                ok.store(false, std::memory_order_relaxed);
              } catch (...) {
                ok.store(false, std::memory_order_relaxed);
              }
            });
      }
    } else {
      for (size_t index = 0; index < groups_.size(); ++index) {
        if (!reduceScatterGroup(groups_[index], input, localOutput,
                                elements, elementSize, reduce,
                                groupStats[index])) {
          ok.store(false, std::memory_order_relaxed);
          break;
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "peel_allreduce: group setup or creation failed: "
              << error.what() << "\n";
    ok.store(false, std::memory_order_relaxed);
  } catch (...) {
    ok.store(false, std::memory_order_relaxed);
  }
  // Drain every successfully created worker even if another group or thread
  // creation failed. No buffer/reference can outlive this operation, and the
  // complete allreduce cannot enter allgather before this join barrier.
  for (auto& thread : groupThreads) {
    thread.join();
  }
  for (const auto& result : groupStats) {
    stats_.reduceScatterSendBytes += result.reduceScatterSendBytes;
    stats_.reduceScatterBroadcasts += result.reduceScatterBroadcasts;
  }
  return ok.load(std::memory_order_relaxed);
}

bool PeelAllreduce::reduceScatterGroup(
    OwnerGroup& group, const void* input, void* localOutput,
    size_t elements, size_t elementSize, const ReduceFunction& reduce,
    Stats& groupStats) {
  groupStats = Stats{};
  try {
    const int world = config_.world_size;
    const size_t shard = shardSize(elements, world);
    const size_t tileLimit = tileBytes_ / elementSize;
    const auto* inputBytes = static_cast<const uint8_t*>(input);
    auto* outputBytes = static_cast<uint8_t*>(localOutput);
    const bool owner = config_.rank >= group.first &&
        config_.rank < group.first + group.count;
    if (validCount(elements, shard, group.first, 0, shard) == 0) {
      return true;
    }
    std::vector<AlignedBuffer> received(world);
    const size_t maximumTransfer = static_cast<size_t>(group.count) *
        std::min(tileLimit, shard) * elementSize;
    if (owner) {
      for (auto& buffer : received) {
        buffer.resize(maximumTransfer);
      }
    } else {
      received[config_.rank].resize(maximumTransfer);
    }
    for (size_t offset = 0; offset < shard;) {
      const size_t count = std::min(tileLimit, shard - offset);
      const size_t tileSize = count * elementSize;
      const size_t transferSize = static_cast<size_t>(group.count) * tileSize;
      if (validCount(elements, shard, group.first, offset, count) == 0) {
        offset += count;
        continue;
      }
      auto* own = static_cast<uint8_t*>(received[config_.rank].data());
      std::memset(own, 0, transferSize);
      for (int slot = 0; slot < group.count; ++slot) {
        const int shardOwner = group.first + slot;
        const size_t valid = validCount(elements, shard, shardOwner, offset, count);
        if (valid != 0) {
          const size_t position = static_cast<size_t>(shardOwner) * shard + offset;
          std::memcpy(own + slot * tileSize,
                      inputBytes + position * elementSize, valid * elementSize);
        }
      }
      std::atomic<bool> ok{true};
      std::vector<std::thread> threads;
      threads.reserve(world);
      try {
        for (int root = 0; root < world; ++root) {
          if (!group.sources[root]) {
            continue;
          }
          ++groupStats.reduceScatterBroadcasts;
          if (root == config_.rank) {
            groupStats.reduceScatterSendBytes += transferSize;
          }
          threads.emplace_back([&group, &received, &ok, root, transferSize]() {
            try {
              if (!group.sources[root]->broadcast(0, received[root].data(), transferSize)) {
                ok.store(false, std::memory_order_relaxed);
              }
            } catch (const std::exception& error) {
              std::cerr << "peel_allreduce: broadcast exception: " << error.what() << "\n";
              ok.store(false, std::memory_order_relaxed);
            } catch (...) {
              ok.store(false, std::memory_order_relaxed);
            }
          });
        }
      } catch (const std::exception& error) {
        std::cerr << "peel_allreduce: worker creation failed: " << error.what() << "\n";
        ok.store(false, std::memory_order_relaxed);
      } catch (...) {
        ok.store(false, std::memory_order_relaxed);
      }
      for (auto& thread : threads) {
        thread.join();
      }
      if (!ok.load(std::memory_order_relaxed)) {
        return false;
      }
      // Every source has finished reading this tile before an owner writes its
      // in-place result. Concurrent groups access disjoint input shard ranges;
      // only this rank's single owner group writes its local output shard.
      if (owner) {
        const size_t valid = validCount(elements, shard, config_.rank, offset, count);
        if (valid != 0) {
          const int slot = config_.rank - group.first;
          auto* destination = outputBytes + offset * elementSize;
          const auto* first = static_cast<const uint8_t*>(received[0].data()) + slot * tileSize;
          std::memcpy(destination, first, valid * elementSize);
          for (int root = 1; root < world; ++root) {
            const auto* contribution = static_cast<const uint8_t*>(received[root].data()) + slot * tileSize;
            reduce(destination, destination, contribution, valid);
          }
        }
      }
      offset += count;
    }
    return true;
  } catch (const std::exception& error) {
    // Allocations occur before source workers start; callback exceptions occur
    // after those workers have joined. Inner creation failures drain workers.
    std::cerr << "peel_allreduce: group exception: " << error.what() << "\n";
  } catch (...) {
  }
  return false;
}

bool PeelAllreduce::gatherShards(void* data, size_t elements, size_t elementSize) {
  const int world = config_.world_size;
  if (world == 1 || elements == 0) {
    return true;
  }
  const size_t shard = shardSize(elements, world);
  const size_t tileLimit = tileBytes_ / elementSize;
  auto* dataBytes = static_cast<uint8_t*>(data);
  std::vector<AlignedBuffer> received(world);
  std::vector<void*> buffers;
  buffers.reserve(world);
  for (auto& buffer : received) {
    buffer.resize(std::min(tileLimit, shard) * elementSize);
    buffers.push_back(buffer.data());
  }
  for (size_t offset = 0; offset < shard;) {
    const size_t count = std::min(tileLimit, shard - offset);
    const size_t tileSize = count * elementSize;
    auto* own = static_cast<uint8_t*>(buffers[config_.rank]);
    std::memset(own, 0, tileSize);
    const size_t valid = validCount(elements, shard, config_.rank, offset, count);
    if (valid != 0) {
      const size_t position = static_cast<size_t>(config_.rank) * shard + offset;
      std::memcpy(own, dataBytes + position * elementSize, valid * elementSize);
    }
    int active = 0;
    while (active < world && validCount(elements, shard, active, offset, count) != 0) {
      ++active;
    }
    stats_.allgatherBroadcasts += active;
    if (config_.rank < active) {
      stats_.allgatherSendBytes += tileSize;
    }
    if (active == world) {
      if (!gather_->run(buffers, tileSize)) {
        return false;
      }
    } else {
      // Nonempty shards are a prefix of global ranks. Keep global root IDs
      // intact by selecting precisely that context/buffer prefix; each cached
      // context still broadcasts to the full world, including empty owners.
      std::vector<PeelContext*> activeContexts;
      std::vector<void*> activeBuffers;
      for (int root = 0; root < active; ++root) {
        activeContexts.push_back(gatherContexts_[root].get());
        activeBuffers.push_back(buffers[root]);
      }
      PeelAllgather activeGather(std::move(activeContexts), PeelAllgatherMode::Parallel);
      if (!activeGather.run(activeBuffers, tileSize)) {
        return false;
      }
    }
    for (int owner = 0; owner < world; ++owner) {
      const size_t validOwner = validCount(elements, shard, owner, offset, count);
      if (validOwner != 0) {
        const size_t position = static_cast<size_t>(owner) * shard + offset;
        std::memcpy(dataBytes + position * elementSize, buffers[owner],
                    validOwner * elementSize);
      }
    }
    offset += count;
  }
  return true;
}

bool PeelAllreduce::run(
    void* data, size_t elements, size_t elementSize, const ReduceFunction& reduce) {
  stats_ = Stats{};
  if (!checkArguments(data, elements, elementSize, reduce)) {
    return false;
  }
  if (elements == 0) {
    return true;
  }
  const size_t shard = shardSize(elements, config_.world_size);
  const size_t start = std::min(elements, static_cast<size_t>(config_.rank) * shard);
  auto* localOutput = static_cast<uint8_t*>(data) + start * elementSize;
  if (!reduceScatter(data, localOutput, elements, elementSize, reduce)) {
    return false;
  }
  const auto startTime = Clock::now();
  const bool ok = gatherShards(data, elements, elementSize);
  stats_.allgatherSeconds =
      std::chrono::duration<double>(Clock::now() - startTime).count();
  return ok;
}

} // namespace peel
} // namespace transport
} // namespace gloo

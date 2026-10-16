#pragma once

#include <arpa/inet.h>

#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "gloo/barrier.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/common/logging.h"
#include "gloo/transport/peel/peel_allreduce.h"
#include "gloo/transport/peel/peel_allreduce_ring_sharded.h"
#include "gloo/transport/peel/peel_context.h"

namespace gloo {
namespace benchmark {

// Real benchmark runner adapter for the two new PEEL allreduce algorithms.
// TCP is used only by the runner's setup and batch-control operations. Each
// timed run below resets a bounded input and carries its collective via PEEL.
template <typename T>
class PeelAllreduceRenoBenchmark : public Benchmark<T> {
 public:
  PeelAllreduceRenoBenchmark(
      std::shared_ptr<::gloo::Context>& context,
      options& opts,
      bool sdn)
      : Benchmark<T>(context, opts), sdn_(sdn) {}

  void initialize(size_t elements) override {
    const auto& opts = this->options_;
    GLOO_ENFORCE_EQ(opts.transport, "peel", "use --transport=peel");
    GLOO_ENFORCE_EQ(opts.threads, 1, "PEEL collectives require --threads=1");
    GLOO_ENFORCE_EQ(opts.inputs, 1, "these allreduce selectors require --inputs=1");
    GLOO_ENFORCE_GT(opts.iterationCount, 0, "specify --iteration-count");
    GLOO_ENFORCE_GT(opts.peelTimeoutMs, 0, "--peel-timeout must be positive");
    GLOO_ENFORCE_GT(opts.peelRtoMs, 0, "--peel-rto must be positive");
    GLOO_ENFORCE_GE(opts.peelMaxPayload, 0, "--peel-max-payload must be nonnegative");
    GLOO_ENFORCE_GT(opts.peelTileBytes, 0, "--peel-tile-bytes must be positive");
    GLOO_ENFORCE_GE(opts.peelTileBytes, sizeof(T), "--peel-tile-bytes is too small");
    GLOO_ENFORCE_EQ(opts.peelTileBytes % sizeof(T), 0, "tile must align to the dtype");
    GLOO_ENFORCE(!opts.peelIface.empty(), "specify --peel-iface");
    GLOO_ENFORCE_EQ(opts.peelHosts.size(), this->context_->size,
                    "--peel-hosts must contain one IPv4 address per rank");
    GLOO_ENFORCE(sdn_ || !opts.peelReduceParallel,
                  "--peel-reduce-parallel applies only to native SDN allreduce");
    if (sdn_) {
      GLOO_ENFORCE(!opts.peelTopologyFile.empty(), "SDN allreduce requires --peel-topology-file");
    }

    transport::peel::PeelContextConfig cfg;
    cfg.rank = this->context_->rank;
    cfg.world_size = this->context_->size;
    std::unordered_set<std::string> unique;
    for (int rank = 0; rank < cfg.world_size; ++rank) {
      in_addr address{};
      const auto& ip = opts.peelHosts[rank];
      GLOO_ENFORCE_EQ(inet_pton(AF_INET, ip.c_str(), &address), 1, "invalid IPv4 address: ", ip);
      GLOO_ENFORCE(unique.insert(ip).second, "duplicate --peel-hosts address: ", ip);
      cfg.peer_ips[rank] = ip;
    }
    const auto span = transport::peel::PeelAllreduce::portSpan(cfg.world_size);
    GLOO_ENFORCE_GT(opts.peelBasePort, 0, "base port must be positive");
    GLOO_ENFORCE_LE(opts.peelBasePort, 65535, "base port exceeds UDP range");
    GLOO_ENFORCE_GT(span, 0, "unsupported PEEL world size");
    GLOO_ENFORCE_LE(span, 65536 - static_cast<uint64_t>(opts.peelBasePort),
                    "base port plus reserved PEEL span exceeds UDP range");
    cfg.base_port = static_cast<uint16_t>(opts.peelBasePort);
    cfg.iface_name = opts.peelIface;
    cfg.topology_file = opts.peelTopologyFile;
    cfg.mcast_group = opts.peelMcastGroup;
    cfg.ttl = opts.peelTTL;
    cfg.timeout_ms = opts.peelTimeoutMs;
    cfg.rto_ms = opts.peelRtoMs;
    cfg.max_chunk_size = static_cast<size_t>(opts.peelMaxPayload);
    cfg.reno_dupack_pct = opts.peelRenoDupackPct;
    cfg.reno_tagg_ms = opts.peelRenoTaggMs;
    cfg.reno_ooo_buffer_segments = static_cast<uint32_t>(opts.peelRenoOooBuf);
    cfg.reno_rto_reset_on_ack = opts.peelRenoRtoResetOnAck;
    elements_ = elements;
    this->allocate(1, elements);
    if (sdn_) {
      native_ = std::make_unique<transport::peel::PeelAllreduce>(cfg, opts.peelTileBytes);
      native_->setReduceGroupsParallel(opts.peelReduceParallel);
      GLOO_ENFORCE(native_->initReno(), "SDN PEEL allreduce initialization failed");
    } else {
      ringContext_ = std::make_unique<transport::peel::PeelContext>(cfg);
      GLOO_ENFORCE(ringContext_->initRingReno(), "PEEL Reno ring initialization failed");
      ring_ = std::make_unique<transport::peel::PeelAllreduceRingShardedGeneric>(
          cfg.rank, ringContext_->ringHops());
      GLOO_ENFORCE(ring_->isReady(), "sharded ring is not ready");
    }
    // All raw sockets must be initialized before any PEEL data transfer.
    BarrierOptions barrierOptions(this->context_);
    barrierOptions.setTag(0xBADC0DE5);
    barrier(barrierOptions);
  }

  void run() override {
    auto& input = this->inputs_[0];
    // Adjacent calls change input to help expose stale DATA; generation wraps
    // every 31 calls. N8 sums remain exactly representable in float16.
    lastGeneration_ = callGeneration_++ % 31;
    for (size_t i = 0; i < elements_; ++i) {
      input[i] = T(this->context_->rank + (i % 17) + lastGeneration_);
    }
    const auto sum = [](void* output, const void* a, const void* b, size_t n) {
      auto* dst = static_cast<T*>(output);
      const auto* lhs = static_cast<const T*>(a);
      const auto* rhs = static_cast<const T*>(b);
      for (size_t i = 0; i < n; ++i) {
        dst[i] = lhs[i] + rhs[i];
      }
    };
    const bool ok = sdn_
        ? native_->run(input.data(), elements_, sizeof(T), sum)
        : ring_->run(input.data(), elements_, sizeof(T), sum);
    GLOO_ENFORCE(ok, "PEEL allreduce failed");
  }

  void verify(std::vector<std::string>&) override {
    const auto world = this->context_->size;
    for (size_t i = 0; i < elements_; ++i) {
      const T expected(world * ((i % 17) + lastGeneration_) + world * (world - 1) / 2);
      GLOO_ENFORCE_EQ(this->inputs_[0][i], expected, "PEEL allreduce mismatch at ", i);
    }
    std::cout << "PEEL_VERIFY rank=" << this->context_->rank
              << " elements=" << elements_ << " dtype_bytes=" << sizeof(T)
              << " generation=" << lastGeneration_
              << " backend=" << (sdn_ ? "sdn_reno" : "ring_sharded_reno")
              << " reduce_parallel=" << (this->options_.peelReduceParallel ? "true" : "false")
              << " rs_send_bytes=" << (sdn_ ? native_->lastStats().reduceScatterSendBytes : ring_->lastStats().reduceScatterSendBytes)
              << " ag_send_bytes=" << (sdn_ ? native_->lastStats().allgatherSendBytes : ring_->lastStats().allgatherSendBytes)
              << " status=pass" << std::endl;
  }

 private:
  bool sdn_;
  size_t elements_ = 0;
  size_t callGeneration_ = 0;
  size_t lastGeneration_ = 0;
  std::unique_ptr<transport::peel::PeelAllreduce> native_;
  std::unique_ptr<transport::peel::PeelContext> ringContext_;
  std::unique_ptr<transport::peel::PeelAllreduceRingShardedGeneric> ring_;
};

} // namespace benchmark
} // namespace gloo

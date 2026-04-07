#include "tb/Analysis/KernelContract.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::tb;

namespace mlir::tb {
#define GEN_PASS_DEF_TBVERIFYPIPELINEMAINLINECONTRACT
#include "tb/Transforms/Passes.h.inc"

namespace {

struct Position {
  int64_t stage = -1;
  int64_t cluster = -1;
  int64_t order = -1;
};

struct ExplicitPipelineCluster {
  ExpandedClusterKind kind = ExpandedClusterKind::AsyncIssue;
  int64_t ordinal = -1;
  int64_t stage = -1;
  int64_t cluster = -1;
  int64_t kGroup = -1;
  SmallVector<int64_t, 32> opIds;
  SmallVector<int64_t, 8> waitGroupIds;
  bool needsBarrier = false;
};

static bool isStrictlyEarlier(Position lhs, Position rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order < rhs.order;
}

static SmallVector<int64_t, 8> copyI64Array(ArrayRef<int64_t> values) {
  return SmallVector<int64_t, 8>(values.begin(), values.end());
}

static ExplicitPipelineCluster buildExplicitCluster(AsyncIssueClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::AsyncIssue;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  return cluster;
}

static ExplicitPipelineCluster buildExplicitCluster(ConsumerWaitClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::ConsumerWait;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  cluster.waitGroupIds = copyI64Array(op.getWaitGroupIds());
  cluster.needsBarrier = op.getNeedsBarrier();
  return cluster;
}

static ExplicitPipelineCluster buildExplicitCluster(MmaComputeClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::MmaCompute;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  return cluster;
}

static FailureOr<SmallVector<ExplicitPipelineCluster, 8>>
collectExplicitPipelineClusters(PipelineMainlineOp op) {
  if (op.getBody().empty() || !llvm::hasSingleElement(op.getBody())) {
    op.emitError()
        << "pipeline_mainline must contain a single explicit pipeline body";
    return failure();
  }

  SmallVector<ExplicitPipelineCluster, 8> clusters;
  for (Operation &nestedOp : op.getBody().front()) {
    if (auto async = dyn_cast<AsyncIssueClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(async));
      continue;
    }
    if (auto wait = dyn_cast<ConsumerWaitClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(wait));
      continue;
    }
    if (auto mma = dyn_cast<MmaComputeClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(mma));
      continue;
    }
    op.emitError() << "unsupported op in explicit pipeline body: "
                   << nestedOp.getName().getStringRef();
    return failure();
  }

  if (clusters.empty()) {
    op.emitError() << "pipeline_mainline must contain explicit pipeline clusters";
    return failure();
  }
  return clusters;
}

static int64_t findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords,
                                  StringRef axis) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  return it == coords.end() ? -1 : it->value;
}

static LogicalResult
validateEpilogueWorkspaceContract(const KernelContract &contract,
                                  PipelineMainlineOp op) {
  const TargetLandingPlan &landing = contract.epilogue.targetLanding;
  if (landing.kind != TargetLandingKind::RegisterPackGlobalVector) {
    return op.emitError()
           << "strict pipeline contract requires final C landing to stay "
              "register-pack/global-vector";
  }
  if (landing.sharedTileRows != 0 || landing.sharedTileCols != 0 ||
      landing.sharedPackSlots != 0 || landing.initSharedStoreVectorWidth != 0 ||
      landing.initSharedLoadVectorWidth != 0 ||
      landing.storeSharedStoreVectorWidth != 0 ||
      landing.storeSharedLoadVectorWidth != 0 ||
      landing.useSharedPackForInit || landing.useSharedPackForStore ||
      landing.requiredSharedBytes != 0 || landing.requiredSyncKind != "none") {
    return op.emitError()
           << "target landing still carries shared scratch metadata after the "
              "reorder/workspace split";
  }
  if (contract.epilogueReorder.kind == EpilogueReorderKind::None)
    return success();
  if (contract.epilogueReorder.kind !=
          EpilogueReorderKind::CTASharedRowReorder ||
      contract.sharedWorkspace.selectedPolicy.empty() ||
      contract.resourceClosure.selectedSharedWorkspacePolicy.empty()) {
    return op.emitError()
           << "pipeline contract requires explicit CTA shared row reorder + "
              "shared workspace closure";
  }
  if (failed(findSharedWorkspaceSegment(
          contract.sharedWorkspace,
          SharedWorkspaceSegmentKind::EpilogueReorderScratch,
          "epilogue_init_reorder_scratch", op.getOperation())) ||
      failed(findSharedWorkspaceSegment(
          contract.sharedWorkspace,
          SharedWorkspaceSegmentKind::EpilogueReorderScratch,
          "epilogue_store_reorder_scratch", op.getOperation()))) {
    return failure();
  }
  if (contract.resourceClosure.workspaceTotalBytes !=
          contract.sharedWorkspace.totalBytes ||
      contract.resourceClosure.peakStaticSharedBytes !=
          contract.sharedWorkspace.peakBytes) {
    return op.emitError()
           << "resource closure must match the shared workspace owner truth";
  }
  return success();
}

class TBVerifyPipelineMainlineContract
    : public impl::TBVerifyPipelineMainlineContractBase<
          TBVerifyPipelineMainlineContract> {
public:
  using impl::TBVerifyPipelineMainlineContractBase<
      TBVerifyPipelineMainlineContract>::TBVerifyPipelineMainlineContractBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool hadFailure = false;

    module.walk([&](PipelineMainlineOp op) {
      auto contract = parseKernelContract(op.getOperation());
      auto explicitClusters = collectExplicitPipelineClusters(op);
      if (failed(contract) || failed(explicitClusters)) {
        hadFailure = true;
        return;
      }
      if (failed(validateEpilogueWorkspaceContract(*contract, op))) {
        hadFailure = true;
        return;
      }

      DenseMap<int64_t, const PipelineOp *> opsById;
      DenseMap<int64_t, const ValueState *> valuesById;
      DenseMap<int64_t, Position> positionByOp;
      DenseMap<int64_t, const AsyncGroup *> groupById;
      DenseMap<int64_t, const WaitInfo *> waitByGroupId;
      DenseMap<int64_t, const AsyncProducer *> cpAsyncProducerByValue;
      DenseSet<int64_t> explicitOps;
      DenseSet<int64_t> explicitWaitedGroups;
      DenseSet<int64_t> waitedGroups;
      int64_t maxScheduledStage = -1;

      for (const PipelineOp &pipelineOp : contract->buffers.ops)
        opsById[pipelineOp.id] = &pipelineOp;
      for (const ValueState &value : contract->buffers.values)
        valuesById[value.id] = &value;

      for (const auto &it : llvm::enumerate(*explicitClusters)) {
        const ExplicitPipelineCluster &cluster = it.value();
        maxScheduledStage = std::max(maxScheduledStage, cluster.stage);
        if (cluster.ordinal != static_cast<int64_t>(it.index())) {
          op.emitError()
              << "explicit pipeline clusters must carry contiguous ordinals";
          hadFailure = true;
          return;
        }
        if (cluster.opIds.empty()) {
          op.emitError() << "explicit pipeline clusters must not be empty";
          hadFailure = true;
          return;
        }
        for (const auto &orderedOp : llvm::enumerate(cluster.opIds)) {
          int64_t opId = orderedOp.value();
          auto opIt = opsById.find(opId);
          if (opIt == opsById.end()) {
            op.emitError()
                << "explicit pipeline cluster references unknown op " << opId;
            hadFailure = true;
            return;
          }
          if (!explicitOps.insert(opId).second) {
            op.emitError() << "explicit pipeline clusters duplicate op " << opId;
            hadFailure = true;
            return;
          }
          positionByOp[opId] = Position{cluster.stage, cluster.cluster,
                                        static_cast<int64_t>(orderedOp.index())};
          int64_t opKGroup =
              findIterationCoord(opIt->second->iterationCoords, "k_group");
          if (opKGroup != cluster.kGroup) {
            op.emitError() << "explicit pipeline cluster k_group "
                           << cluster.kGroup << " disagrees with op " << opId
                           << " k_group " << opKGroup;
            hadFailure = true;
            return;
          }

          switch (cluster.kind) {
          case ExpandedClusterKind::AsyncIssue:
            if (opIt->second->kind != BufferOpKind::LoadA &&
                opIt->second->kind != BufferOpKind::LoadB) {
              op.emitError()
                  << "async_issue_cluster may only contain load_a/load_b ops";
              hadFailure = true;
              return;
            }
            break;
          case ExpandedClusterKind::ConsumerWait:
            if (opIt->second->kind != BufferOpKind::LocalLoadA &&
                opIt->second->kind != BufferOpKind::LocalLoadB) {
              op.emitError() << "consumer_wait_cluster may only contain "
                                "local_load_a/local_load_b ops";
              hadFailure = true;
              return;
            }
            break;
          case ExpandedClusterKind::MmaCompute:
            if (opIt->second->kind != BufferOpKind::Mma) {
              op.emitError()
                  << "mma_compute_cluster may only contain mma ops";
              hadFailure = true;
              return;
            }
            break;
          }
        }
      }

      if (explicitOps.size() != contract->buffers.ops.size()) {
        op.emitError()
            << "explicit pipeline clusters must cover every pipeline op exactly once";
        hadFailure = true;
        return;
      }
      if (contract->pipelineReady.scheduledMaxStage != maxScheduledStage) {
        op.emitError() << "pipeline_ready stage summary disagrees with explicit "
                          "pipeline clusters";
        hadFailure = true;
        return;
      }

      for (const AsyncGroup &group : contract->async.groups) {
        if (!groupById.try_emplace(group.id, &group).second) {
          op.emitError() << "duplicate async group " << group.id;
          hadFailure = true;
          return;
        }
      }
      for (const AsyncProducer &producer : contract->async.producers) {
        if (producer.kind != AsyncProducerKind::CpAsync)
          continue;
        if (!producer.legal) {
          op.emitError()
              << "pipeline contract verifier does not accept illegal cp.async producers";
          hadFailure = true;
          return;
        }
        if (!cpAsyncProducerByValue.try_emplace(producer.valueId, &producer)
                 .second) {
          op.emitError() << "duplicate cp.async producer for value "
                         << producer.valueId;
          hadFailure = true;
          return;
        }
        if (!groupById.count(producer.groupId)) {
          op.emitError() << "cp.async producer references missing group "
                         << producer.groupId;
          hadFailure = true;
          return;
        }
      }

      for (const WaitInfo &wait : contract->async.waits) {
        if (!waitByGroupId.try_emplace(wait.groupId, &wait).second) {
          op.emitError() << "duplicate wait metadata for async group "
                         << wait.groupId;
          hadFailure = true;
          return;
        }
        if (!groupById.count(wait.groupId)) {
          op.emitError() << "wait references unknown async group " << wait.groupId;
          hadFailure = true;
          return;
        }
        if (!waitedGroups.insert(wait.groupId).second) {
          op.emitError() << "duplicate wait for async group " << wait.groupId;
          hadFailure = true;
          return;
        }
        const AsyncGroup *group = groupById.lookup(wait.groupId);
        if (!group || group->producers.size() != 1) {
          op.emitError()
              << "pipeline contract verifier expects one producer per async group";
          hadFailure = true;
          return;
        }
        int64_t producerIndex = group->producers.front();
        if (producerIndex < 0 ||
            producerIndex >= static_cast<int64_t>(contract->async.producers.size())) {
          op.emitError() << "async group " << wait.groupId
                         << " references invalid producer index";
          hadFailure = true;
          return;
        }
        const AsyncProducer &producer = contract->async.producers[producerIndex];
        auto valueIt = valuesById.find(producer.valueId);
        auto usePosIt = positionByOp.find(wait.beforeOpId);
        auto defPosIt = positionByOp.find(producer.opId);
        if (valueIt == valuesById.end() || usePosIt == positionByOp.end() ||
            defPosIt == positionByOp.end()) {
          op.emitError() << "wait must anchor to a scheduled pipeline value";
          hadFailure = true;
          return;
        }
        if (!llvm::is_contained(valueIt->second->users, wait.beforeOpId)) {
          op.emitError() << "wait before_op " << wait.beforeOpId
                         << " is not a user of value " << producer.valueId;
          hadFailure = true;
          return;
        }
        Position required{wait.requiredStage, wait.requiredCluster,
                          wait.requiredOrder};
        Position actualUse = usePosIt->second;
        if (required.stage != actualUse.stage ||
            required.cluster != actualUse.cluster ||
            required.order != actualUse.order) {
          op.emitError() << "wait frontier for value " << producer.valueId
                         << " does not match the scheduled first-use position";
          hadFailure = true;
          return;
        }
        if (!isStrictlyEarlier(defPosIt->second, actualUse)) {
          op.emitError() << "waited value " << producer.valueId
                         << " must be defined before its first use";
          hadFailure = true;
          return;
        }
      }

      for (const ExplicitPipelineCluster &cluster : *explicitClusters) {
        if (cluster.kind == ExpandedClusterKind::ConsumerWait &&
            cluster.waitGroupIds.empty()) {
          op.emitError()
              << "consumer_wait_cluster must carry wait ownership";
          hadFailure = true;
          return;
        }
        if (cluster.kind != ExpandedClusterKind::ConsumerWait &&
            !cluster.waitGroupIds.empty()) {
          op.emitError()
              << "only consumer_wait_cluster may carry wait ownership";
          hadFailure = true;
          return;
        }
        if (cluster.kind == ExpandedClusterKind::ConsumerWait &&
            !cluster.needsBarrier) {
          op.emitError()
              << "consumer_wait_cluster must carry CTA barrier ownership";
          hadFailure = true;
          return;
        }
        if (cluster.kind != ExpandedClusterKind::ConsumerWait &&
            cluster.needsBarrier) {
          op.emitError()
              << "non-consumer explicit clusters must not request CTA barrier";
          hadFailure = true;
          return;
        }

        for (int64_t groupId : cluster.waitGroupIds) {
          if (!explicitWaitedGroups.insert(groupId).second) {
            op.emitError()
                << "explicit pipeline clusters duplicate waited group "
                << groupId;
            hadFailure = true;
            return;
          }
          auto waitIt = waitByGroupId.find(groupId);
          if (waitIt == waitByGroupId.end()) {
            op.emitError()
                << "explicit pipeline clusters reference unknown wait group "
                << groupId;
            hadFailure = true;
            return;
          }
          const WaitInfo *wait = waitIt->second;
          if (wait->requiredStage != cluster.stage ||
              wait->requiredCluster != cluster.cluster) {
            op.emitError() << "explicit pipeline cluster does not match wait "
                              "frontier for group "
                           << groupId;
            hadFailure = true;
            return;
          }
          if (!cluster.needsBarrier && wait->needsBarrier) {
            op.emitError()
                << "explicit pipeline cluster dropped the required CTA barrier for "
                   "group "
                << groupId;
            hadFailure = true;
            return;
          }
        }
      }

      if (explicitWaitedGroups.size() != contract->async.waits.size()) {
        op.emitError() << "explicit pipeline clusters must cover every async "
                          "wait exactly once";
        hadFailure = true;
        return;
      }
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

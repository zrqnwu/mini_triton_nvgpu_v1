#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/EpilogueReorderPlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/Analysis/PipelineExpansion.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/Analysis/PipelineReady.h"
#include "tb/Analysis/ResourceClosurePlan.h"
#include "tb/Analysis/SharedWorkspacePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBCLEANUPPIPELINE
#include "tb/Transforms/Passes.h.inc"

namespace {

static void materializeAsyncCluster(OpBuilder &builder, Location loc,
                                    const ExpandedCluster &cluster) {
  AsyncIssueClusterOp::create(
      builder, loc,
      builder.getI64IntegerAttr(cluster.ordinal),
      builder.getI64IntegerAttr(cluster.stage),
      builder.getI64IntegerAttr(cluster.cluster),
      builder.getI64IntegerAttr(cluster.kGroup),
      builder.getDenseI64ArrayAttr(cluster.opIds),
      builder.getStringAttr(cluster.reason));
}

static void materializeWaitCluster(OpBuilder &builder, Location loc,
                                   const ExpandedCluster &cluster) {
  ConsumerWaitClusterOp::create(
      builder, loc, builder.getI64IntegerAttr(cluster.ordinal),
      builder.getI64IntegerAttr(cluster.stage),
      builder.getI64IntegerAttr(cluster.cluster),
      builder.getI64IntegerAttr(cluster.kGroup),
      builder.getDenseI64ArrayAttr(cluster.opIds),
      builder.getDenseI64ArrayAttr(cluster.waitGroupIds),
      builder.getBoolAttr(cluster.needsBarrier),
      builder.getStringAttr(cluster.reason));
}

static void materializeMmaCluster(OpBuilder &builder, Location loc,
                                  const ExpandedCluster &cluster) {
  MmaComputeClusterOp::create(
      builder, loc, builder.getI64IntegerAttr(cluster.ordinal),
      builder.getI64IntegerAttr(cluster.stage),
      builder.getI64IntegerAttr(cluster.cluster),
      builder.getI64IntegerAttr(cluster.kGroup),
      builder.getDenseI64ArrayAttr(cluster.opIds),
      builder.getStringAttr(cluster.reason));
}

static LogicalResult
validateCleanupEpilogueContract(const KernelConfig &config,
                                const EpiloguePlan &epilogue,
                                const EpilogueReorderPlan &reorder,
                                const SharedWorkspacePlan &workspace,
                                const ResourceClosurePlan &resource,
                                Operation *op) {
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (epilogue.initMode != AccumulatorInitMode::DirectGlobalVector ||
      epilogue.storeMode != AccumulatorStoreMode::DirectGlobalVector || !init ||
      !store) {
    return op->emitError()
           << "pipeline cleanup currently requires explicit "
              "direct_global_vector init/store payloads plus a target landing";
  }

  const TargetLandingPlan &landing = epilogue.targetLanding;
  if (landing.kind != TargetLandingKind::RegisterPackGlobalVector) {
    return op->emitError()
           << "pipeline cleanup now requires final C landing to stay "
              "register-pack/global-vector; shared reorder ownership must live "
              "in tb.epilogue_reorder_plan";
  }
  if (landing.globalVectorWidth <= 0 || landing.directPackRows <= 0 ||
      landing.directPackCols <= 0 || landing.requiredSyncKind.empty()) {
    return op->emitError()
           << "pipeline cleanup requires positive C landing geometry";
  }
  if (landing.globalVectorWidth != init->vectorWidth ||
      landing.globalVectorWidth != store->vectorWidth) {
    return op->emitError()
           << "pipeline cleanup requires init/store direct vector widths to "
              "match the target landing";
  }

  bool boundaryAwareDirect = init->boundaryAware || store->boundaryAware;
  if (config.exactTile) {
    if (landing.kind != TargetLandingKind::RegisterPackGlobalVector ||
        boundaryAwareDirect) {
      return op->emitError()
             << "exact-tile cleanup requires the register-pack direct-global "
                "landing mainline without boundary-aware fallback";
    }
  } else if (landing.kind == TargetLandingKind::RegisterPackGlobalVector &&
             !boundaryAwareDirect) {
    if (reorder.kind == EpilogueReorderKind::None) {
      return op->emitError()
             << "non-exact direct C landing must be either boundary-aware or "
                "explicitly paired with tb.epilogue_reorder_plan";
    }
  }

  if (landing.sharedTileRows != 0 || landing.sharedTileCols != 0 ||
      landing.sharedPackSlots != 0 || landing.initSharedStoreVectorWidth != 0 ||
      landing.initSharedLoadVectorWidth != 0 ||
      landing.storeSharedStoreVectorWidth != 0 ||
      landing.storeSharedLoadVectorWidth != 0 ||
      landing.useSharedPackForInit || landing.useSharedPackForStore ||
      landing.requiredSharedBytes != 0 || landing.requiredSyncKind != "none") {
    return op->emitError()
           << "register-pack direct landing must not carry shared-pack "
              "materialization metadata after reorder/workspace split";
  }

  if (reorder.kind == EpilogueReorderKind::None)
    return success();

  if (reorder.kind != EpilogueReorderKind::CTASharedRowReorder ||
      reorder.sharedTileRows <= 0 || reorder.sharedTileCols <= 0 ||
      reorder.liveSlots <= 0 || reorder.initSharedStoreVectorWidth <= 0 ||
      reorder.initSharedLoadVectorWidth <= 0 ||
      reorder.storeSharedStoreVectorWidth <= 0 ||
      reorder.storeSharedLoadVectorWidth <= 0 ||
      reorder.workspaceBarrierCount <= 0 || !reorder.requiresWarpSync ||
      !reorder.reorderNeededForInit || !reorder.reorderNeededForStore ||
      reorder.workspaceSyncKind != "cta") {
    return op->emitError()
           << "active epilogue reorder must carry complete CTA-workspace owner truth";
  }
  if (reorder.sharedTileCols % landing.globalVectorWidth != 0 ||
      reorder.sharedTileCols % reorder.initSharedLoadVectorWidth != 0 ||
      reorder.sharedTileCols % reorder.storeSharedStoreVectorWidth != 0) {
    return op->emitError()
           << "epilogue reorder tile must stay aligned with shared/global "
              "vector widths";
  }
  if (failed(findSharedWorkspaceSegment(
          workspace, SharedWorkspaceSegmentKind::EpilogueReorderScratch,
          "epilogue_init_reorder_scratch", op)) ||
      failed(findSharedWorkspaceSegment(
          workspace, SharedWorkspaceSegmentKind::EpilogueReorderScratch,
          "epilogue_store_reorder_scratch", op))) {
    return failure();
  }
  if (resource.peakStaticSharedBytes != workspace.peakBytes ||
      resource.workspaceTotalBytes != workspace.totalBytes ||
      resource.selectedSharedWorkspacePolicy != workspace.selectedPolicy) {
    return op->emitError()
           << "resource closure must consume the unified shared workspace truth";
  }
  return success();
}

class TBCleanupPipeline
    : public impl::TBCleanupPipelineBase<TBCleanupPipeline> {
public:
  using impl::TBCleanupPipelineBase<
      TBCleanupPipeline>::TBCleanupPipelineBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    bool hadFailure = false;
    SmallVector<Operation *> toErase;

    module.walk([&](MatmulOp op) {
      auto config = getKernelConfig(op);
      if (failed(config) ||
          failed(verifySupportedKernelConfig(*config, op.getOperation()))) {
        hadFailure = true;
        return;
      }

      auto model = parseBufferModelAttr(op.getOperation());
      auto latency = parseLatencyPlanAttr(op.getOperation());
      auto loopPlan = parseLoopPlanAttr(op.getOperation());
      auto pipeline = parsePipelinePlanAttr(op.getOperation());
      auto async = parseAsyncPlanAttr(op.getOperation());
      auto expansion = parsePipelineExpansionAttr(op.getOperation());
      auto epilogue = parseEpiloguePlanAttr(op.getOperation());
      auto epilogueReorder = parseEpilogueReorderPlanAttr(op.getOperation());
      auto sharedWorkspace = parseSharedWorkspacePlanAttr(op.getOperation());
      auto resource = parseResourceClosurePlanAttr(op.getOperation());
      if (failed(model) || failed(latency) || failed(loopPlan) ||
          failed(pipeline) || failed(async) || failed(expansion) ||
          failed(epilogue) || failed(epilogueReorder) ||
          failed(sharedWorkspace) || failed(resource)) {
        hadFailure = true;
        return;
      }
      (void)latency;

      if (failed(
              validateCleanupEpilogueContract(*config, *epilogue,
                                              *epilogueReorder,
                                              *sharedWorkspace, *resource,
                                              op.getOperation()))) {
        hadFailure = true;
        return;
      }

      DenseSet<int64_t> asyncLoadOps;
      for (const AsyncProducer &producer : async->producers) {
        if (producer.kind != AsyncProducerKind::CpAsync || !producer.legal) {
          op.emitError()
              << "cleanup requires every async producer to be legal cp.async";
          hadFailure = true;
          return;
        }
        asyncLoadOps.insert(producer.opId);
      }
      if (async->groups.size() != async->waits.size() ||
          async->groups.size() != async->producers.size()) {
        op.emitError()
            << "cleanup requires one async group and one wait per producer";
        hadFailure = true;
        return;
      }

      for (const PipelineOp &pipelineOp : model->ops) {
        if (pipelineOp.kind != BufferOpKind::LoadA &&
            pipelineOp.kind != BufferOpKind::LoadB) {
          continue;
        }
        if (!asyncLoadOps.count(pipelineOp.id)) {
          op.emitError() << "cleanup found a scheduled shared load without "
                            "cp.async ownership";
          hadFailure = true;
          return;
        }
      }

      DenseSet<int64_t> expandedOps;
      DenseSet<int64_t> expandedWaitGroups;
      DenseSet<int64_t> seenKGroups;
      SmallVector<const ExpandedCluster *, 16> orderedClusters;
      int64_t maxKGroup = -1;
      for (const ExpandedCluster &cluster : expansion->clusters) {
        for (int64_t opId : cluster.opIds)
          expandedOps.insert(opId);
        for (int64_t groupId : cluster.waitGroupIds)
          expandedWaitGroups.insert(groupId);
        if (cluster.kGroup < 0) {
          op.emitError() << "cleanup requires pipeline_expansion to carry an "
                            "explicit non-negative `k_group` owner";
          hadFailure = true;
          return;
        }
        orderedClusters.push_back(&cluster);
        maxKGroup = std::max(maxKGroup, cluster.kGroup);
        seenKGroups.insert(cluster.kGroup);
      }
      if (expandedOps.size() != model->ops.size()) {
        op.emitError() << "cleanup requires pipeline_expansion to cover every "
                          "pipeline op exactly once";
        hadFailure = true;
        return;
      }
      if (expandedWaitGroups.size() != async->waits.size()) {
        op.emitError() << "cleanup requires pipeline_expansion to own every "
                          "async wait exactly once";
        hadFailure = true;
        return;
      }

      int64_t numKGroups = maxKGroup + 1;
      if (numKGroups != loopPlan->iterationCount) {
        op.emitError() << "pipeline_expansion k_group coverage disagrees with "
                          "tb.loop_plan iteration_count";
        hadFailure = true;
        return;
      }
      for (int64_t kGroup = 0; kGroup < numKGroups; ++kGroup) {
        if (!seenKGroups.count(kGroup)) {
          op.emitError() << "cleanup requires contiguous k_group ownership for "
                            "explicit pipeline clusters";
          hadFailure = true;
          return;
        }
      }

      PipelineReady ready;
      ready.scheduledMaxStage = pipeline->scheduledMaxStage;
      ready.asyncGroups = async->groups.size();
      ready.requestedStages = config->requestedStages;

      OpBuilder opBuilder(op);
      auto pipelineMainline =
          PipelineMainlineOp::create(opBuilder, op.getLoc(), op.getA(),
                                     op.getB(), op.getC());

      for (const NamedAttribute &attr : op->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name == "block_m" || name == "block_n" || name == "block_k" ||
            name == "num_warps" || name == "num_stages" ||
            name == "group_m" ||
            name == "exact_tile" || name == "mma" ||
            name == "tb.target_info" ||
            name == "tb.layout_plan" || name == "tb.c_register_plan" ||
            name == "tb.mainloop_graph" || name == "tb.schedule_plan" ||
            name == "tb.wait_plan" ||
            name == "tb.loop_plan" || name == "tb.latency_plan" ||
            name == "tb.pipeline_plan" || name == "tb.pipeline_expansion" ||
            name == "tb.pipeline_ready") {
          continue;
        }
        pipelineMainline->setAttr(name, attr.getValue());
      }
      pipelineMainline->setAttr("tb.pipeline_ready",
                                buildPipelineReadyAttr(builder, ready));

      Block *body = new Block();
      pipelineMainline.getBody().push_back(body);
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
      for (const ExpandedCluster *cluster : orderedClusters) {
        switch (cluster->kind) {
        case ExpandedClusterKind::AsyncIssue:
          materializeAsyncCluster(bodyBuilder, op.getLoc(), *cluster);
          break;
        case ExpandedClusterKind::ConsumerWait:
          materializeWaitCluster(bodyBuilder, op.getLoc(), *cluster);
          break;
        case ExpandedClusterKind::MmaCompute:
          materializeMmaCluster(bodyBuilder, op.getLoc(), *cluster);
          break;
        }
      }
      toErase.push_back(op.getOperation());
    });

    for (Operation *op : toErase)
      op->erase();

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

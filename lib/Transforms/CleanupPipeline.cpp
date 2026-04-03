#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/Analysis/PipelineExpansion.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/Analysis/PipelineReady.h"
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
      if (failed(model) || failed(latency) || failed(loopPlan) ||
          failed(pipeline) || failed(async) || failed(expansion) ||
          failed(epilogue)) {
        hadFailure = true;
        return;
      }
      (void)latency;

      if (epilogue->initMode != AccumulatorInitMode::DirectGlobalVector ||
          epilogue->storeMode != AccumulatorStoreMode::DirectGlobalVector) {
        op.emitError() << "pipeline cleanup only accepts direct-global epilogues";
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

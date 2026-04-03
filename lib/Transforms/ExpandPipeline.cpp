#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/PipelineExpansion.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBEXPANDPIPELINE
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBExpandPipeline
    : public impl::TBExpandPipelineBase<TBExpandPipeline> {
public:
  using impl::TBExpandPipelineBase<TBExpandPipeline>::TBExpandPipelineBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    bool hadFailure = false;

    module.walk([&](MatmulOp op) {
      auto config = getKernelConfig(op);
      if (failed(config) ||
          failed(verifySupportedKernelConfig(*config, op.getOperation()))) {
        hadFailure = true;
        return;
      }

      auto model = parseBufferModelAttr(op.getOperation());
      auto pipelinePlan = parsePipelinePlanAttr(op.getOperation());
      auto asyncPlan = parseAsyncPlanAttr(op.getOperation());
      if (failed(model) || failed(pipelinePlan) || failed(asyncPlan)) {
        hadFailure = true;
        return;
      }

      auto expansion =
          derivePipelineExpansion(*model, *pipelinePlan, *asyncPlan,
                                  op.getOperation());
      if (failed(expansion)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.pipeline_expansion");
      op->setAttr("tb.pipeline_expansion",
                  buildPipelineExpansionAttr(builder, *expansion));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

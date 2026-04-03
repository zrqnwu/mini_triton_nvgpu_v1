#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBDERIVEWAITS
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBDeriveWaits : public impl::TBDeriveWaitsBase<TBDeriveWaits> {
public:
  using impl::TBDeriveWaitsBase<TBDeriveWaits>::TBDeriveWaitsBase;

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
      auto transport = parseTransportPlanAttr(op.getOperation());
      auto latencyPlan = parseLatencyPlanAttr(op.getOperation());
      auto pipelinePlan = parsePipelinePlanAttr(op.getOperation());
      if (failed(model) || failed(transport) || failed(latencyPlan) ||
          failed(pipelinePlan)) {
        hadFailure = true;
        return;
      }

      auto asyncPlan = deriveAsyncPlan(*model, *transport, *latencyPlan,
                                       *pipelinePlan, op.getOperation());
      if (failed(asyncPlan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.wait_plan");
      op->setAttr("tb.async_plan", buildAsyncPlanAttr(builder, *asyncPlan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulRewritePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBBUILDMAINLOOPGRAPH
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBBuildMainloopGraph
    : public impl::TBBuildMainloopGraphBase<TBBuildMainloopGraph> {
public:
  using impl::TBBuildMainloopGraphBase<
      TBBuildMainloopGraph>::TBBuildMainloopGraphBase;

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

      auto encodings = parseEncodingPlanAttr(op.getOperation());
      auto accumulator = parseAccumulatorPlanAttr(op.getOperation());
      auto epilogue = parseEpiloguePlanAttr(op.getOperation());
      auto rewrite = parseMatmulRewritePlanAttr(op.getOperation());
      if (failed(encodings) || failed(accumulator) || failed(epilogue) ||
          failed(rewrite)) {
        hadFailure = true;
        return;
      }

      auto model = deriveBufferModel(*config, *encodings, *accumulator,
                                     *epilogue,
                                     *rewrite, op.getOperation());
      if (failed(model)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.mainloop_graph");
      op->setAttr("tb.buffer_model", buildBufferModelAttr(builder, *model));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulRewritePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBREWRITEMATMULMAINLOOP
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBRewriteMatmulMainloop
    : public impl::TBRewriteMatmulMainloopBase<TBRewriteMatmulMainloop> {
public:
  using impl::TBRewriteMatmulMainloopBase<
      TBRewriteMatmulMainloop>::TBRewriteMatmulMainloopBase;

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
      if (failed(encodings) || failed(accumulator) || failed(epilogue)) {
        hadFailure = true;
        return;
      }

      auto rewrite = deriveMatmulRewritePlan(*config, *encodings, *accumulator,
                                             *epilogue, op.getOperation());
      if (failed(rewrite)) {
        hadFailure = true;
        return;
      }

      op->setAttr("tb.matmul_rewrite",
                  buildMatmulRewritePlanAttr(builder, *rewrite));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

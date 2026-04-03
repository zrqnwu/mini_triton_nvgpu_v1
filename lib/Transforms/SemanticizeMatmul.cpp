#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulSemantics.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBSEMANTICIZEMATMUL
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBSemanticizeMatmul
    : public impl::TBSemanticizeMatmulBase<TBSemanticizeMatmul> {
public:
  using impl::TBSemanticizeMatmulBase<
      TBSemanticizeMatmul>::TBSemanticizeMatmulBase;

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

      auto semantics = deriveMatmulSemantics(*config, op.getOperation());
      if (failed(semantics)) {
        hadFailure = true;
        return;
      }

      op->setAttr("tb.semantic_matmul",
                  buildMatmulSemanticsAttr(builder, *semantics));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

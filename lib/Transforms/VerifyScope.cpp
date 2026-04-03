#include "tb/Analysis/KernelConfig.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBVERIFYSCOPE
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBVerifyScope : public impl::TBVerifyScopeBase<TBVerifyScope> {
public:
  using impl::TBVerifyScopeBase<TBVerifyScope>::TBVerifyScopeBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool hadFailure = false;
    module.walk([&](MatmulOp op) {
      auto config = getKernelConfig(op);
      if (mlir::failed(config) ||
          mlir::failed(
              verifySupportedKernelConfig(*config, op.getOperation()))) {
        hadFailure = true;
      }
    });
    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb

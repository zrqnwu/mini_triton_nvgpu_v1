#ifndef MINI_TRITON_TB_TRANSFORMS_PASSES_H
#define MINI_TRITON_TB_TRANSFORMS_PASSES_H

#include "tb/IR/TBDialect.h"
#include "tb/IR/TBOps.h"

#include "mlir/Pass/Pass.h"

namespace mlir::tb {

void registerStage1PipelineBuilders();

#define GEN_PASS_DECL
#include "tb/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tb/Transforms/Passes.h.inc"

} // namespace mlir::tb

#endif // MINI_TRITON_TB_TRANSFORMS_PASSES_H

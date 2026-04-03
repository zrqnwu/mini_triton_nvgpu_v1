#ifndef MINI_TRITON_TB_ANALYSIS_PIPELINEREADY_H
#define MINI_TRITON_TB_ANALYSIS_PIPELINEREADY_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tb {

struct PipelineReady {
  int64_t scheduledMaxStage = 0;
  int64_t asyncGroups = 0;
  int64_t requestedStages = 0;
};

DictionaryAttr buildPipelineReadyAttr(Builder &builder,
                                      const PipelineReady &ready);
FailureOr<PipelineReady> parsePipelineReadyAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_PIPELINEREADY_H

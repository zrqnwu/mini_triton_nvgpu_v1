#ifndef MINI_TRITON_TB_ANALYSIS_PIPELINEPLAN_H
#define MINI_TRITON_TB_ANALYSIS_PIPELINEPLAN_H

#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/LoopPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct PipelinePlacement {
  int64_t opId = -1;
  int64_t stage = 0;
  int64_t cluster = 0;
  int64_t order = 0;
  std::string reason;
};

struct StageBufferUse {
  int64_t viewId = -1;
  int64_t backing = -1;
  int64_t stage = -1;
  int64_t bufferIndex = -1;
  int64_t producerOp = -1;
  int64_t firstConsumerOp = -1;
  int64_t lastConsumerOp = -1;
};

struct PipelinePlan {
  int64_t scheduledMaxStage = 0;
  llvm::SmallVector<PipelinePlacement, 64> placements;
  llvm::SmallVector<StageBufferUse, 16> stageOwnedBuffers;
};

FailureOr<PipelinePlan> derivePipelinePlan(const BufferModel &model,
                                           const LoopPlan &loopPlan,
                                           const LatencyPlan &latencyPlan,
                                           Operation *op);
DictionaryAttr buildPipelinePlanAttr(Builder &builder,
                                     const PipelinePlan &plan);
FailureOr<PipelinePlan> parsePipelinePlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_PIPELINEPLAN_H

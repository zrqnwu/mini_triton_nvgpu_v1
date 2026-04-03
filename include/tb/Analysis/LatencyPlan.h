#ifndef MINI_TRITON_TB_ANALYSIS_LATENCYPLAN_H
#define MINI_TRITON_TB_ANALYSIS_LATENCYPLAN_H

#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/Analysis/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct OpLatencyInfo {
  int64_t opId = -1;
  int64_t targetLatency = 0;
  int64_t selfLatency = 0;
  int64_t bufferDistance = 0;
  bool pipelineable = false;
  bool accMultiBuffer = false;
  std::string reason;
};

struct LatencyPlan {
  llvm::SmallVector<OpLatencyInfo, 32> ops;
};

FailureOr<LatencyPlan> deriveLatencyPlan(const KernelConfig &config,
                                         const TargetInfo &target,
                                         const BufferModel &model,
                                         const LoopPlan &loopPlan,
                                         Operation *op);
DictionaryAttr buildLatencyPlanAttr(Builder &builder, const LatencyPlan &plan);
FailureOr<LatencyPlan> parseLatencyPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_LATENCYPLAN_H

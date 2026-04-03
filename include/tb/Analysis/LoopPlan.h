#ifndef MINI_TRITON_TB_ANALYSIS_LOOPPLAN_H
#define MINI_TRITON_TB_ANALYSIS_LOOPPLAN_H

#include "tb/Analysis/BufferModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct LoopIterationPlan {
  int64_t kGroup = -1;
  llvm::SmallVector<int64_t, 8> asyncProducerOps;
  llvm::SmallVector<int64_t, 16> consumerOps;
  llvm::SmallVector<int64_t, 32> computeOps;
};

struct LoopCarriedValue {
  int64_t valueId = -1;
  int64_t definingOp = -1;
  int64_t ownerView = -1;
  int64_t loopDistance = 0;
  llvm::SmallVector<int64_t, 8> users;
  std::string reason;
};

struct LoopPlan {
  std::string loopAxis;
  int64_t iterationCount = 0;
  bool singleMainLoop = false;
  llvm::SmallVector<LoopIterationPlan, 8> iterations;
  llvm::SmallVector<LoopCarriedValue, 32> carriedValues;
};

FailureOr<LoopPlan> deriveLoopPlan(const BufferModel &model, Operation *op);
DictionaryAttr buildLoopPlanAttr(Builder &builder, const LoopPlan &plan);
FailureOr<LoopPlan> parseLoopPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_LOOPPLAN_H

#ifndef MINI_TRITON_TB_ANALYSIS_ACCUMULATORPLAN_H
#define MINI_TRITON_TB_ANALYSIS_ACCUMULATORPLAN_H

#include "tb/Analysis/EncodingPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct LaneAccessPattern {
  int64_t laneRowGroupSize = 0;
  int64_t laneColGroupSize = 0;
  int64_t laneColStride = 0;
  llvm::SmallVector<int64_t, 4> rowOffsets;
};

struct AccumulatorPack {
  int64_t packId = 0;
  int64_t rowBase = 0;
  int64_t colBase = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t elemCount = 0;
  int64_t vectorWidth = 0;
  int64_t warpOwner = 0;
  int64_t mmaGroup = -1;
  int64_t epiloguePack = -1;
};

struct AccumulatorPlan {
  int encoding = -1;
  int64_t registersPerWarp = 0;
  std::string ownerScope = "per_warp_template";
  LaneAccessPattern laneAccess;
  llvm::SmallVector<AccumulatorPack, 16> packs;
  bool liveAcrossStages = false;
  int64_t multiBufferDepth = 1;
};

FailureOr<AccumulatorPlan> deriveAccumulatorPlan(const KernelConfig &config,
                                                 const TargetInfo &target,
                                                 const EncodingPlan &encodings,
                                                 Operation *op);
DictionaryAttr buildAccumulatorPlanAttr(Builder &builder,
                                        const AccumulatorPlan &plan);
FailureOr<AccumulatorPlan> parseAccumulatorPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_ACCUMULATORPLAN_H

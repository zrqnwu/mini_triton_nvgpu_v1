#ifndef MINI_TRITON_TB_ANALYSIS_REDUCTIONPLAN_H
#define MINI_TRITON_TB_ANALYSIS_REDUCTIONPLAN_H

#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/ProgramMappingPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tb {

enum class ReductionPlanKind {
  None,
  SplitKSerial,
  SplitKAtomic,
};

enum class ReductionScratchSpace {
  None,
  Global,
  Shared,
};

// 中文标记：split-k 需要单独 owner，不能只留一个 split_k 数字。
struct ReductionPlan {
  ReductionPlanKind kind = ReductionPlanKind::None;
  ReductionScratchSpace scratchSpace = ReductionScratchSpace::None;
  int64_t splitK = 1;
  int64_t kTilesPerProgram = 0;
  int64_t remainderKTiles = 0;
  int64_t partialTileRows = 0;
  int64_t partialTileCols = 0;
  int64_t producerProgramsPerTile = 1;
  int64_t finalReducerPrograms = 1;
  int64_t scratchRows = 0;
  int64_t scratchCols = 0;
  int64_t scratchBytes = 0;
  bool requiresInterProgramReduction = false;
  std::string partialOwner = "none";
  std::string finalExecutor = "none";
  std::string reason;
};

FailureOr<ReductionPlan>
deriveReductionPlan(const KernelConfig &config,
                    const ProgramMappingPlan &programMapping, Operation *op);
DictionaryAttr buildReductionPlanAttr(Builder &builder,
                                      const ReductionPlan &plan);
FailureOr<ReductionPlan> parseReductionPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_REDUCTIONPLAN_H

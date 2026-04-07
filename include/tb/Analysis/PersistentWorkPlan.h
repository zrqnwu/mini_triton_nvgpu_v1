#ifndef MINI_TRITON_TB_ANALYSIS_PERSISTENTWORKPLAN_H
#define MINI_TRITON_TB_ANALYSIS_PERSISTENTWORKPLAN_H

#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tb {

enum class PersistentWorkKind {
  None,
  TileStrideLoop,
};

// 中文标记：persistent 不是一个 bool，而是一份 tile-fetch work owner。
struct PersistentWorkPlan {
  PersistentWorkKind kind = PersistentWorkKind::None;
  bool enabled = false;
  int64_t residentPrograms = 1;
  int64_t programsPerWave = 1;
  int64_t tileBatchSize = 1;
  int64_t maxTilesPerProgram = 1;
  int64_t totalWorkItems = 1;
  int64_t loopStride = 1;
  bool requiresOuterSerialLoop = false;
  std::string tileOrder = "row_major";
  std::string ownerScope = "none";
  std::string completion = "single_tile";
  std::string reason;
};

FailureOr<PersistentWorkPlan>
derivePersistentWorkPlan(const KernelConfig &config, const TargetInfo &target,
                         const ProgramMappingPlan &programMapping, Operation *op);
DictionaryAttr buildPersistentWorkPlanAttr(Builder &builder,
                                           const PersistentWorkPlan &plan);
FailureOr<PersistentWorkPlan> parsePersistentWorkPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_PERSISTENTWORKPLAN_H

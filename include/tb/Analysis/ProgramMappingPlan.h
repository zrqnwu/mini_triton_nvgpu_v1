#ifndef MINI_TRITON_TB_ANALYSIS_PROGRAMMAPPINGPLAN_H
#define MINI_TRITON_TB_ANALYSIS_PROGRAMMAPPINGPLAN_H

#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tb {

enum class ProgramMappingKind {
  Tile,
  GroupedTile,
  SplitK,
  PersistentTile,
};

enum class ProgramLaunchOrder {
  RowMajor,
  GroupedM,
  Persistent,
};

enum class ProgramSwizzleKind {
  None,
  GroupedM,
};

enum class ReductionMode {
  None,
  SplitKSerial,
  SplitKParallel,
};

// 中文标记：这里只表达 CTA / program 级映射，不表达 CTA 内部 pipeline。
struct ProgramMappingPlan {
  ProgramMappingKind mappingKind = ProgramMappingKind::Tile;
  int64_t problemTilesM = 0;
  int64_t problemTilesN = 0;
  int64_t problemTilesK = 0;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  int64_t groupM = 1;
  // 中文标记：grouped/tile launch 的显式公式组件。
  // lowering 只能消费这些字段，不再自己重新拼 launch 公式常量。
  int64_t groupTileSpanM = 1;
  int64_t groupTileSpanN = 1;
  // 中文标记：这里表示一个 launch group 的最大 pid 宽度，不等于真实 launch 总块数。
  int64_t programsPerLaunchGroup = 1;
  int64_t launchGroupCount = 1;
  // 中文标记：这里表示真实 launch 的 program 总数；grouped_tile 的尾组可小于
  // programsPerLaunchGroup，所以 totalPrograms 不能再由简单乘法假定。
  int64_t totalPrograms = 1;
  int64_t splitK = 1;
  ProgramLaunchOrder launchOrder = ProgramLaunchOrder::RowMajor;
  ProgramSwizzleKind swizzleKind = ProgramSwizzleKind::None;
  bool persistent = false;
  ReductionMode reductionMode = ReductionMode::None;
  int64_t programsPerTile = 1;
  int64_t numCTAs = 1;
  llvm::SmallVector<int64_t, 3> ctasPerCGA = {1, 1, 1};
  llvm::SmallVector<int64_t, 3> ctaSplitNum = {1, 1, 1};
  llvm::SmallVector<int64_t, 3> ctaOrder = {0, 1, 2};
};

FailureOr<ProgramMappingPlan>
deriveProgramMappingPlan(const KernelConfig &config, const TargetInfo &target,
                         Operation *op);
DictionaryAttr buildProgramMappingPlanAttr(Builder &builder,
                                           const ProgramMappingPlan &plan);
FailureOr<ProgramMappingPlan> parseProgramMappingPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_PROGRAMMAPPINGPLAN_H

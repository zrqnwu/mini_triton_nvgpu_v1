#ifndef MINI_TRITON_TB_ANALYSIS_SHAREDWORKSPACEPLAN_H
#define MINI_TRITON_TB_ANALYSIS_SHAREDWORKSPACEPLAN_H

#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpilogueReorderPlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/PersistentWorkPlan.h"
#include "tb/Analysis/ReductionPlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/Analysis/TransportPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::tb {

enum class SharedWorkspaceSegmentKind {
  MainloopAStageBuffer,
  MainloopBStageBuffer,
  EpilogueReorderScratch,
  SplitKReductionScratch,
  PersistentStateScratch,
};

struct SharedWorkspaceSegment {
  SharedWorkspaceSegmentKind kind =
      SharedWorkspaceSegmentKind::MainloopAStageBuffer;
  std::string name;
  int64_t byteOffset = 0;
  int64_t byteSize = 0;
  int64_t byteAlignment = 1;
  int64_t logicalRows = 0;
  int64_t logicalCols = 0;
  int64_t stageCount = 1;
  int64_t slotCount = 1;
  int64_t warpReplicas = 1;
  int64_t lifetimeBegin = 0;
  int64_t lifetimeEnd = 0;
  bool aliasAllowed = false;
  int64_t aliasSet = -1;
  std::string producer;
  std::string consumer;
};

struct SharedWorkspacePlan {
  std::string contractModel;
  int64_t totalBytes = 0;
  int64_t peakBytes = 0;
  int64_t aliasSavedBytes = 0;
  std::string selectedPolicy;
  std::string reason;
  llvm::SmallVector<SharedWorkspaceSegment, 8> segments;
};

FailureOr<SharedWorkspacePlan>
deriveSharedWorkspacePlan(const KernelConfig &config, const TargetInfo &target,
                          const EncodingPlan &encodings,
                          const TransportPlan &transport,
                          const EpilogueReorderPlan &reorder,
                          const ReductionPlan &reduction,
                          const PersistentWorkPlan &persistentWork,
                          Operation *op);
DictionaryAttr buildSharedWorkspacePlanAttr(Builder &builder,
                                            const SharedWorkspacePlan &plan);
FailureOr<SharedWorkspacePlan> parseSharedWorkspacePlanAttr(Operation *op);
FailureOr<const SharedWorkspaceSegment *>
findSharedWorkspaceSegment(const SharedWorkspacePlan &plan,
                           SharedWorkspaceSegmentKind kind,
                           StringRef name,
                           Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_SHAREDWORKSPACEPLAN_H

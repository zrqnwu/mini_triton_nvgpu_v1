#ifndef MINI_TRITON_TB_ANALYSIS_PIPELINEEXPANSION_H
#define MINI_TRITON_TB_ANALYSIS_PIPELINEEXPANSION_H

#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/PipelinePlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

enum class ExpandedClusterKind {
  AsyncIssue,
  ConsumerWait,
  MmaCompute,
};

struct ExpandedCluster {
  int64_t ordinal = -1;
  int64_t stage = -1;
  int64_t cluster = -1;
  // 中文标记：k_group 必须在 pipeline expand 时就成为 cluster owner。
  // cleanup/lowering 只能消费它，不能再从 op_id 反推。
  int64_t kGroup = -1;
  ExpandedClusterKind kind = ExpandedClusterKind::AsyncIssue;
  llvm::SmallVector<int64_t, 32> opIds;
  llvm::SmallVector<int64_t, 8> waitGroupIds;
  // 中文标记：这是 pipeline expand 之后的显式 CTA 同步真相。
  // lowering 只能消费这个字段，不能再按 stage/cluster 自己重新聚 wait。
  bool needsBarrier = false;
  std::string reason;
};

struct PipelineExpansion {
  llvm::SmallVector<ExpandedCluster, 16> clusters;
};

FailureOr<PipelineExpansion>
derivePipelineExpansion(const BufferModel &model, const PipelinePlan &pipelinePlan,
                        const AsyncPlan &asyncPlan, Operation *op);
DictionaryAttr buildPipelineExpansionAttr(Builder &builder,
                                          const PipelineExpansion &expansion);
FailureOr<PipelineExpansion> parsePipelineExpansionAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_PIPELINEEXPANSION_H

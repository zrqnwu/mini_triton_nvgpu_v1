#ifndef MINI_TRITON_TB_ANALYSIS_ASYNCPLAN_H
#define MINI_TRITON_TB_ANALYSIS_ASYNCPLAN_H

#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/Analysis/TransportPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

enum class AsyncProducerKind {
  CpAsync,
  TMA,
  BulkCopy,
  SyncCopyFallback,
};

enum class AsyncBarrierKind {
  AsyncGroup,
  CTA,
  MBarrier,
  None,
};

enum class AsyncCachePolicy {
  Default,
  BypassL1,
  CacheAll,
};

struct AsyncProducer {
  int64_t opId = -1;
  AsyncProducerKind kind = AsyncProducerKind::CpAsync;
  int64_t valueId = -1;
  int64_t srcView = -1;
  int64_t dstView = -1;
  // 中文标记：global->shared 的源 tile 偏移必须在 async 合同里显式拥有，
  // 不能让 lowering 再按 k_group/shape 去猜。
  llvm::SmallVector<int64_t, 4> srcOffsets;
  int64_t groupId = -1;
  int64_t vecBytes = 0;
  // 中文标记：下面这些字段是 transport 扩展位，当前 strict matmul 主线
  // 仍然只真正使用 cp.async。
  int64_t transactionBytes = 0;
  bool zeroFill = false;
  bool predicated = false;
  bool bypassL1 = false;
  AsyncBarrierKind barrierKind = AsyncBarrierKind::AsyncGroup;
  AsyncCachePolicy cachePolicy = AsyncCachePolicy::Default;
  bool legal = false;
  std::string reason;
};

struct AsyncGroup {
  int64_t id = -1;
  llvm::SmallVector<int64_t, 8> producers;
};

struct WaitInfo {
  int64_t groupId = -1;
  int64_t beforeOpId = -1;
  int64_t requiredStage = -1;
  int64_t requiredCluster = -1;
  int64_t requiredOrder = -1;
  bool needsBarrier = true;
  std::string reason;
};

struct ReuseFence {
  int64_t viewId = -1;
  int64_t backing = -1;
  int64_t retiringValueId = -1;
  int64_t acquiringValueId = -1;
  int64_t afterOpId = -1;
  int64_t requiredAfterStage = -1;
  int64_t requiredAfterCluster = -1;
  int64_t requiredAfterOrder = -1;
  std::string reason;
};

struct AsyncPlan {
  llvm::SmallVector<AsyncProducer, 16> producers;
  llvm::SmallVector<AsyncGroup, 16> groups;
  llvm::SmallVector<WaitInfo, 16> waits;
  llvm::SmallVector<ReuseFence, 16> reuseFences;
};

FailureOr<AsyncPlan> deriveAsyncPlan(const BufferModel &model,
                                     const TransportPlan &transport,
                                     const LatencyPlan &latencyPlan,
                                     const PipelinePlan &pipelinePlan,
                                     Operation *op);
DictionaryAttr buildAsyncPlanAttr(Builder &builder, const AsyncPlan &plan);
FailureOr<AsyncPlan> parseAsyncPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_ASYNCPLAN_H

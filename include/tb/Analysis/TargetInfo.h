#ifndef MINI_TRITON_TB_ANALYSIS_TARGETINFO_H
#define MINI_TRITON_TB_ANALYSIS_TARGETINFO_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

inline constexpr char kTBTargetModuleAttrName[] = "tb.target";
inline constexpr char kTBTargetOpAttrName[] = "tb.target_info";
inline constexpr char kTBNumWarpsAttrName[] = "tb.num-warps";
inline constexpr char kTBThreadsPerWarpAttrName[] =
    "tb.threads-per-warp";
inline constexpr char kTBNumCTAsAttrName[] = "tb.num-ctas";
inline constexpr char kTBRequestedStagesAttrName[] = "tb.requested-stages";

struct TargetInfo {
  std::string gpuArch;
  std::string targetTriple;
  std::string ptxFeatures;
  int64_t threadsPerWarp = 32;
  int64_t sharedBankBytes = 4;
  int64_t numSms = 0;
  int64_t maxWarpsPerCTA = 0;
  int64_t maxRegistersPerThread = 0;
  int64_t maxRegistersPerCTA = 0;
  int64_t maxSharedBytesPerCTA = 0;
  bool supportsAsyncCopy = true;
  bool supportsLdMatrix = true;
  bool supportsMmaSync = true;
  bool supportsMBarrier = false;
  bool supportsTMA = false;
  bool supportsWGMMA = false;
  int64_t asyncCopyMinBytes = 4;
  int64_t asyncCopyMaxBytes = 16;
  int64_t asyncCopyPreferredBytes = 16;
  int64_t globalToSharedStageLatency = 2;
  int64_t sharedToRegisterStageLatency = 1;
  int64_t mmaStageLatency = 1;
  // 中文标记：这里先只保留 transport 偏好，不把 TargetInfo 变成通用调度器。
  std::string preferredAsyncTransport = "cp_async";
  llvm::SmallVector<int64_t, 4> mmaInstrShape;
};

FailureOr<TargetInfo> deriveTargetInfoForArch(StringRef gpuArch,
                                              StringRef ptxFeatures,
                                              Operation *op);
DictionaryAttr buildTargetInfoAttr(Builder &builder, const TargetInfo &target);
FailureOr<TargetInfo> parseTargetInfoAttr(Operation *op);
FailureOr<TargetInfo> getTargetInfo(Operation *op);
LogicalResult setModuleContextAttr(ModuleOp module, StringRef name,
                                   Attribute value, Operation *anchorOp);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_TARGETINFO_H

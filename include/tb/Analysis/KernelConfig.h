#ifndef MINI_TRITON_TB_ANALYSIS_KERNELCONFIG_H
#define MINI_TRITON_TB_ANALYSIS_KERNELCONFIG_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::tb {

class MatmulOp;
class PipelineMainlineOp;

enum class ScalarKind {
  F16,
  F32,
};

enum class MmaKind {
  M16N8K16,
};

struct KernelConfig {
  int64_t problemM = 0;
  int64_t problemN = 0;
  int64_t problemK = 0;
  int64_t blockM = 0;
  int64_t blockN = 0;
  int64_t blockK = 0;
  int64_t numWarps = 0;
  int64_t requestedStages = 0;
  int64_t groupM = 8;
  bool exactTile = false;
  MmaKind mmaKind = MmaKind::M16N8K16;
  ScalarKind aScalar = ScalarKind::F16;
  ScalarKind bScalar = ScalarKind::F16;
  ScalarKind cScalar = ScalarKind::F32;
};

llvm::StringRef stringifyScalarKind(ScalarKind kind);
llvm::StringRef stringifyMmaKind(MmaKind kind);

FailureOr<KernelConfig> getKernelConfig(MatmulOp op);
FailureOr<KernelConfig> getKernelConfig(PipelineMainlineOp op);
LogicalResult verifySupportedKernelConfig(const KernelConfig &config,
                                         Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_KERNELCONFIG_H

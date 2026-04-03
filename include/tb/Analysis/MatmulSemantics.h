#ifndef MINI_TRITON_TB_ANALYSIS_MATMULSEMANTICS_H
#define MINI_TRITON_TB_ANALYSIS_MATMULSEMANTICS_H

#include "tb/Analysis/KernelConfig.h"
#include "tb/IR/TBTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct MatmulSemantics {
  std::string contractModel;
  mlir::Type aDescType;
  mlir::Type bDescType;
  mlir::Type cDescType;
  int64_t problemM = 0;
  int64_t problemN = 0;
  int64_t problemK = 0;
  int64_t tileM = 0;
  int64_t tileN = 0;
  int64_t tileK = 0;
  bool exactTile = false;
  bool hasBoundaryM = false;
  bool hasBoundaryN = false;
  bool hasBoundaryK = false;
};

FailureOr<MatmulSemantics> deriveMatmulSemantics(const KernelConfig &config,
                                                 Operation *op);
DictionaryAttr buildMatmulSemanticsAttr(Builder &builder,
                                        const MatmulSemantics &semantics);
FailureOr<MatmulSemantics> parseMatmulSemanticsAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_MATMULSEMANTICS_H

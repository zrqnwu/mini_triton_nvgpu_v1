#include "tb/IR/TBAttrs.h"

#include "tb/IR/TBDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

static LogicalResult emitArrayRankError(function_ref<InFlightDiagnostic()> emitError,
                                        StringRef name,
                                        ArrayRef<int64_t> lhs,
                                        ArrayRef<int64_t> rhs) {
  return emitError() << name << " rank mismatch: " << lhs.size() << " vs "
                     << rhs.size();
}

static LogicalResult emitPositiveArrayError(
    function_ref<InFlightDiagnostic()> emitError, StringRef name,
    ArrayRef<int64_t> values) {
  if (llvm::any_of(values, [](int64_t value) { return value <= 0; }))
    return emitError() << name << " entries must be positive";
  return success();
}

static LogicalResult verifyPermutation(
    function_ref<InFlightDiagnostic()> emitError, StringRef name,
    ArrayRef<int64_t> values) {
  SmallVector<bool, 8> seen(values.size(), false);
  for (int64_t value : values) {
    if (value < 0 || value >= static_cast<int64_t>(values.size()))
      return emitError() << name << " must be a permutation of [0, rank)";
    if (seen[value])
      return emitError() << name << " must not contain duplicates";
    seen[value] = true;
  }
  return success();
}

} // namespace

LogicalResult SemanticLayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> order,
    ArrayRef<int64_t> strides, bool zeroOffset, bool hasStaticStrides,
    bool unitStrideOnMinor, bool contiguous) {
  (void)zeroOffset;
  if (order.empty())
    return emitError() << "semantic layout must have non-empty order";
  if (order.size() != strides.size())
    return emitError() << "semantic layout stride rank must match order rank";
  if (failed(verifyPermutation(emitError, "order", order)))
    return failure();
  if (hasStaticStrides) {
    if (llvm::any_of(strides, [](int64_t value) { return value <= 0; }))
      return emitError() << "semantic layout static strides must be positive";
  } else if (llvm::any_of(strides, [](int64_t value) { return value == 0; })) {
    return emitError() << "semantic layout strides must use positive values or -1"
                       << " for dynamic metadata";
  }
  if ((unitStrideOnMinor || contiguous) && !hasStaticStrides) {
    return emitError() << "semantic layout contiguity flags require static "
                          "stride metadata";
  }
  if (contiguous && !unitStrideOnMinor) {
    return emitError() << "semantic layout contiguous=true requires "
                          "unitStrideOnMinor=true";
  }
  return success();
}

LogicalResult CGAEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                      ArrayRef<int64_t> ctasPerCGA,
                                      ArrayRef<int64_t> ctaSplitNum,
                                      ArrayRef<int64_t> ctaOrder) {
  if (ctasPerCGA.size() != 3 || ctaSplitNum.size() != 3 ||
      ctaOrder.size() != 3) {
    return emitError()
           << "cga encoding must carry rank-3 ctas_per_cga/cta_split_num/cta_order";
  }
  if (failed(emitPositiveArrayError(emitError, "ctas_per_cga", ctasPerCGA)) ||
      failed(emitPositiveArrayError(emitError, "cta_split_num", ctaSplitNum)) ||
      failed(verifyPermutation(emitError, "cta_order", ctaOrder))) {
    return failure();
  }
  return success();
}

LogicalResult BlockedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                          ArrayRef<int64_t> sizePerThread,
                                          ArrayRef<int64_t> threadsPerWarp,
                                          ArrayRef<int64_t> warpsPerCTA,
                                          ArrayRef<int64_t> order,
                                          CGAEncodingAttr cga) {
  if (sizePerThread.empty())
    return emitError() << "blocked encoding must have non-empty size_per_thread";
  if (sizePerThread.size() != threadsPerWarp.size())
    return emitArrayRankError(emitError, "size_per_thread/threads_per_warp",
                              sizePerThread, threadsPerWarp);
  if (sizePerThread.size() != warpsPerCTA.size())
    return emitArrayRankError(emitError, "size_per_thread/warps_per_cta",
                              sizePerThread, warpsPerCTA);
  if (sizePerThread.size() != order.size())
    return emitError() << "order rank must match blocked encoding rank";
  if (!cga)
    return emitError() << "blocked encoding requires a cga owner";
  if (failed(emitPositiveArrayError(emitError, "size_per_thread", sizePerThread)) ||
      failed(emitPositiveArrayError(emitError, "threads_per_warp", threadsPerWarp)) ||
      failed(emitPositiveArrayError(emitError, "warps_per_cta", warpsPerCTA)) ||
      failed(verifyPermutation(emitError, "order", order))) {
    return failure();
  }
  return success();
}

LogicalResult SharedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                         ArrayRef<int64_t> order,
                                         int64_t perPhase,
                                         int64_t maxPhase,
                                         ArrayRef<int64_t> paddingIntervals,
                                         ArrayRef<int64_t> paddings,
                                         bool transposed,
                                         int64_t swizzlingByteWidth,
                                         CGAEncodingAttr cga) {
  (void)transposed;
  if (order.empty())
    return emitError() << "shared encoding must have non-empty order";
  if (!cga)
    return emitError() << "shared encoding requires a cga owner";
  if (paddingIntervals.size() != paddings.size())
    return emitError() << "padding_intervals and paddings must have same rank";
  if (!paddingIntervals.empty() && paddingIntervals.size() != order.size())
    return emitError() << "padding rank must match shared encoding rank";
  if (perPhase < 0 || maxPhase < 0)
    return emitError() << "per_phase and max_phase must be non-negative";
  if (swizzlingByteWidth < 0)
    return emitError() << "swizzling_byte_width must be non-negative";
  if (failed(verifyPermutation(emitError, "order", order)))
    return failure();
  if (llvm::any_of(paddingIntervals, [](int64_t value) { return value < 0; }) ||
      llvm::any_of(paddings, [](int64_t value) { return value < 0; })) {
    return emitError() << "shared padding metadata must be non-negative";
  }
  return success();
}

LogicalResult NVGPUMmaEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                           StringRef mmaFamily,
                                           int64_t versionMajor,
                                           int64_t versionMinor,
                                           ArrayRef<int64_t> warpsPerCTA,
                                           ArrayRef<int64_t> instrShape,
                                           CGAEncodingAttr cga) {
  if (mmaFamily.empty())
    return emitError() << "mma_family must be non-empty";
  if (versionMajor < 0 || versionMinor < 0)
    return emitError() << "mma version must be non-negative";
  if (warpsPerCTA.size() != 2)
    return emitError() << "warps_per_cta must have rank 2";
  if (instrShape.size() != 3)
    return emitError() << "instr_shape must have rank 3";
  if (!cga)
    return emitError() << "nvgpu mma encoding requires a cga owner";
  if (failed(emitPositiveArrayError(emitError, "warps_per_cta", warpsPerCTA)) ||
      failed(emitPositiveArrayError(emitError, "instr_shape", instrShape))) {
    return failure();
  }
  return success();
}

LogicalResult DotOperandEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                             int64_t operandIndex,
                                             NVGPUMmaEncodingAttr parent,
                                             int64_t kWidth) {
  (void)parent;
  if (operandIndex != 0 && operandIndex != 1)
    return emitError() << "operand_index must be 0 or 1";
  if (kWidth <= 0)
    return emitError() << "k_width must be positive";
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "tb/IR/TBAttrs.cpp.inc"

void TBDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tb/IR/TBAttrs.cpp.inc"
      >();
}

Attribute TBDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) {
    if (succeeded(*parseResult))
      return attr;
    return {};
  }
  parser.emitError(parser.getNameLoc()) << "unknown tb attribute `" << attrTag
                                        << "`";
  return {};
}

void TBDialect::printAttribute(Attribute attr,
                               DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
  llvm_unreachable("unexpected tb attribute kind");
}

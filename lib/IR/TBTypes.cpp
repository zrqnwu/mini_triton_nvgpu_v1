#include "tb/IR/TBTypes.h"

#include "tb/IR/TBAttrs.h"
#include "tb/IR/TBDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

} // namespace

LogicalResult MemDescType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  Attribute encoding, StringRef memorySpace,
                                  bool mutableMemory,
                                  ArrayRef<int64_t> allocShape) {
  (void)mutableMemory;
  if (shape.empty())
    return emitError() << "shape must be non-empty";
  if (shape.size() != allocShape.size())
    return emitError() << "alloc_shape rank must match shape rank";
  if (llvm::any_of(shape, [](int64_t dim) { return dim <= 0; }))
    return emitError() << "shape dimensions must be positive";
  if (llvm::any_of(allocShape, [](int64_t dim) { return dim <= 0; }))
    return emitError() << "alloc_shape dimensions must be positive";
  if (!elementType)
    return emitError() << "element_type must be present";
  if (memorySpace != "global" && memorySpace != "shared" &&
      memorySpace != "registers") {
    return emitError() << "memory_space must be one of global/shared/registers";
  }

  if (isa<SemanticLayoutAttr>(encoding) && memorySpace != "global")
    return emitError() << "semantic layout requires global memory_space";
  if (isa<SharedEncodingAttr>(encoding) && memorySpace != "shared")
    return emitError() << "shared encoding requires shared memory_space";
  if ((isa<DotOperandEncodingAttr>(encoding) || isa<AccumulatorEncodingAttr>(encoding)) &&
      memorySpace != "registers") {
    return emitError() << "register fragment encodings require registers memory_space";
  }
  return success();
}

#define GET_TYPEDEF_CLASSES
#include "tb/IR/TBTypes.cpp.inc"

void TBDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tb/IR/TBTypes.cpp.inc"
      >();
}

Type TBDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  Type type;
  auto parseResult = generatedTypeParser(parser, &typeTag, type);
  if (parseResult.has_value()) {
    if (succeeded(*parseResult))
      return type;
    return {};
  }
  parser.emitError(parser.getNameLoc()) << "unknown tb type `" << typeTag
                                        << "`";
  return {};
}

void TBDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (succeeded(generatedTypePrinter(type, printer)))
    return;
  llvm_unreachable("unexpected tb type kind");
}

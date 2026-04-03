#include "tb/Analysis/MatmulSemantics.h"

#include "tb/Analysis/BufferModel.h"
#include "tb/IR/TBAttrs.h"

#include "llvm/ADT/STLExtras.h"

#include <iterator>
#include <tuple>

using namespace mlir;
using namespace mlir::tb;

namespace {

static DenseI64ArrayAttr buildI64ArrayAttr(Builder &builder,
                                           ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

static SmallVector<int64_t> parseI64Array(DenseI64ArrayAttr attr) {
  return SmallVector<int64_t>(attr.asArrayRef().begin(), attr.asArrayRef().end());
}

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static FailureOr<bool> readBoolField(DictionaryAttr dict, StringRef name,
                                     Operation *op) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing bool field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static FailureOr<MemDescType> getTypedMemDesc(Type type, Operation *op,
                                              StringRef role) {
  auto desc = dyn_cast<MemDescType>(type);
  if (!desc) {
    op->emitError() << "expected !tb.memdesc for `" << role << "`";
    return failure();
  }
  return desc;
}

struct OperandLayoutTruth {
  SmallVector<int64_t, 2> strides;
  SmallVector<int64_t, 2> order;
  bool zeroOffset = true;
  bool hasStaticStrides = true;
  bool unitStrideOnMinor = false;
  bool contiguous = false;
};

static FailureOr<OperandLayoutTruth>
deriveOperandLayoutTruth(MemRefType type, Operation *op, StringRef role) {
  SmallVector<int64_t, 4> rawStrides;
  int64_t offset = 0;
  if (failed(type.getStridesAndOffset(rawStrides, offset))) {
    op->emitError() << "semantic operand `" << role
                    << "` must carry explicit strided layout metadata";
    return failure();
  }
  if (type.getRank() != 2 || rawStrides.size() != 2) {
    op->emitError() << "semantic operand `" << role
                    << "` must be a rank-2 strided memref";
    return failure();
  }

  OperandLayoutTruth truth;
  truth.zeroOffset = offset == 0;
  truth.hasStaticStrides =
      llvm::all_of(rawStrides, [](int64_t stride) { return stride > 0; });
  truth.strides.reserve(rawStrides.size());
  for (int64_t stride : rawStrides)
    truth.strides.push_back(stride > 0 ? stride : -1);

  auto inferOrderFromStatic = [&]() -> FailureOr<SmallVector<int64_t, 2>> {
    if (!truth.hasStaticStrides)
      return failure();
    if (truth.strides[0] == truth.strides[1]) {
      op->emitError() << "semantic operand `" << role
                      << "` has ambiguous stride order";
      return failure();
    }
    return truth.strides[1] < truth.strides[0]
               ? SmallVector<int64_t, 2>{1, 0}
               : SmallVector<int64_t, 2>{0, 1};
  };

  if (truth.hasStaticStrides) {
    auto order = inferOrderFromStatic();
    if (failed(order))
      return failure();
    truth.order = *order;
    int64_t minor = truth.order.front();
    int64_t major = truth.order.back();
    truth.unitStrideOnMinor = truth.strides[minor] == 1;
    truth.contiguous = truth.unitStrideOnMinor &&
                       truth.strides[major] == type.getShape()[minor];
    return truth;
  }

  auto unitStrideDim = llvm::find_if(rawStrides, [](int64_t stride) {
    return stride == 1;
  });
  if (unitStrideDim == rawStrides.end()) {
    op->emitError() << "semantic operand `" << role
                    << "` currently requires inferable minor order metadata";
    return failure();
  }
  int64_t minor = static_cast<int64_t>(std::distance(rawStrides.begin(), unitStrideDim));
  truth.order = minor == 0 ? SmallVector<int64_t, 2>{0, 1}
                           : SmallVector<int64_t, 2>{1, 0};
  truth.unitStrideOnMinor = true;
  truth.contiguous = false;
  return truth;
}

static LogicalResult validateSemanticLayout(ArrayRef<int64_t> shape,
                                            SemanticLayoutAttr layout,
                                            StringRef role, Operation *op) {
  if (!layout)
    return op->emitError() << "semantic operand `" << role
                           << "` must carry #tb.semantic_layout encoding";
  if (layout.getOrder().size() != 2 || layout.getStrides().size() != 2) {
    return op->emitError() << "semantic operand `" << role
                           << "` must carry rank-2 order/stride metadata";
  }
  if (shape.size() != 2)
    return op->emitError() << "semantic operand `" << role
                           << "` must stay rank-2";
  if (layout.getContiguous() &&
      !layout.getUnitStrideOnMinor()) {
    return op->emitError() << "semantic operand `" << role
                           << "` contiguous semantic layout must keep unit "
                              "stride on the minor dimension";
  }
  return success();
}

static LogicalResult validateMatmulSemantics(const MatmulSemantics &semantics,
                                             Operation *op) {
  if (semantics.contractModel.empty())
    return op->emitError() << "semantic contract model must not be empty";
  if (semantics.problemM <= 0 || semantics.problemN <= 0 ||
      semantics.problemK <= 0) {
    return op->emitError() << "semantic problem M/N/K must be positive";
  }
  if (semantics.tileM <= 0 || semantics.tileN <= 0 || semantics.tileK <= 0)
    return op->emitError() << "semantic tile M/N/K must be positive";
  if (semantics.exactTile &&
      (semantics.hasBoundaryM || semantics.hasBoundaryN ||
       semantics.hasBoundaryK)) {
    return op->emitError() << "semantic exact_tile=true cannot carry boundary "
                              "ownership on M/N/K";
  }

  auto aDesc = getTypedMemDesc(semantics.aDescType, op, "semantic_a");
  auto bDesc = getTypedMemDesc(semantics.bDescType, op, "semantic_b");
  auto cDesc = getTypedMemDesc(semantics.cDescType, op, "semantic_c");
  if (failed(aDesc) || failed(bDesc) || failed(cDesc))
    return failure();

  SmallVector<int64_t, 2> expectedAShape = {semantics.problemM, semantics.problemK};
  SmallVector<int64_t, 2> expectedBShape = {semantics.problemK, semantics.problemN};
  SmallVector<int64_t, 2> expectedCShape = {semantics.problemM, semantics.problemN};
  if (aDesc->getShape() != ArrayRef<int64_t>(expectedAShape))
    return op->emitError() << "semantic A memdesc shape must match problem M/K";
  if (bDesc->getShape() != ArrayRef<int64_t>(expectedBShape))
    return op->emitError() << "semantic B memdesc shape must match problem K/N";
  if (cDesc->getShape() != ArrayRef<int64_t>(expectedCShape))
    return op->emitError() << "semantic C memdesc shape must match problem M/N";

  for (auto item :
       {std::make_tuple(*aDesc, StringRef("a")),
        std::make_tuple(*bDesc, StringRef("b")),
        std::make_tuple(*cDesc, StringRef("c"))}) {
    if (!std::get<0>(item).isGlobalMemory())
      return op->emitError() << "semantic operand `" << std::get<1>(item)
                             << "` must stay in global memory";
    auto layout = dyn_cast<SemanticLayoutAttr>(std::get<0>(item).getEncoding());
    if (failed(validateSemanticLayout(std::get<0>(item).getShape(), layout,
                                      std::get<1>(item), op))) {
      return failure();
    }
  }
  return success();
}

} // namespace

FailureOr<MatmulSemantics> mlir::tb::deriveMatmulSemantics(
    const KernelConfig &config, Operation *op) {
  auto aType = dyn_cast<MemRefType>(op->getOperand(0).getType());
  auto bType = dyn_cast<MemRefType>(op->getOperand(1).getType());
  auto cType = dyn_cast<MemRefType>(op->getOperand(2).getType());
  if (!aType || !bType || !cType) {
    op->emitError() << "semanticization expects memref operands";
    return failure();
  }

  auto aLayout = deriveOperandLayoutTruth(aType, op, "a");
  auto bLayout = deriveOperandLayoutTruth(bType, op, "b");
  auto cLayout = deriveOperandLayoutTruth(cType, op, "c");
  if (failed(aLayout) || failed(bLayout) || failed(cLayout))
    return failure();

  MLIRContext *ctx = op->getContext();
  auto makeSemanticLayout = [&](const OperandLayoutTruth &truth) {
    return SemanticLayoutAttr::get(ctx, truth.order, truth.strides,
                                   truth.zeroOffset, truth.hasStaticStrides,
                                   truth.unitStrideOnMinor, truth.contiguous);
  };

  MatmulSemantics semantics;
  semantics.contractModel = "triton_matmul_semantics_v3";
  semantics.problemM = config.problemM;
  semantics.problemN = config.problemN;
  semantics.problemK = config.problemK;
  semantics.tileM = config.blockM;
  semantics.tileN = config.blockN;
  semantics.tileK = config.blockK;
  semantics.exactTile = config.exactTile;
  semantics.hasBoundaryM = (config.problemM % config.blockM) != 0;
  semantics.hasBoundaryN = (config.problemN % config.blockN) != 0;
  semantics.hasBoundaryK = (config.problemK % config.blockK) != 0;
  semantics.aDescType = MemDescType::get(
      ctx, ArrayRef<int64_t>{config.problemM, config.problemK},
      getTypeForScalarKind(ctx, config.aScalar), makeSemanticLayout(*aLayout),
      "global", /*mutableMemory=*/true,
      ArrayRef<int64_t>{config.problemM, config.problemK});
  semantics.bDescType = MemDescType::get(
      ctx, ArrayRef<int64_t>{config.problemK, config.problemN},
      getTypeForScalarKind(ctx, config.bScalar), makeSemanticLayout(*bLayout),
      "global", /*mutableMemory=*/true,
      ArrayRef<int64_t>{config.problemK, config.problemN});
  semantics.cDescType = MemDescType::get(
      ctx, ArrayRef<int64_t>{config.problemM, config.problemN},
      getTypeForScalarKind(ctx, config.cScalar), makeSemanticLayout(*cLayout),
      "global", /*mutableMemory=*/true,
      ArrayRef<int64_t>{config.problemM, config.problemN});

  if (failed(validateMatmulSemantics(semantics, op)))
    return failure();
  return semantics;
}

DictionaryAttr mlir::tb::buildMatmulSemanticsAttr(
    Builder &builder, const MatmulSemantics &semantics) {
  NamedAttrList attrs;
  attrs.set("contract_model", builder.getStringAttr(semantics.contractModel));
  attrs.set("a_desc_type", TypeAttr::get(semantics.aDescType));
  attrs.set("b_desc_type", TypeAttr::get(semantics.bDescType));
  attrs.set("c_desc_type", TypeAttr::get(semantics.cDescType));
  attrs.set("problem_shape",
            buildI64ArrayAttr(builder, {semantics.problemM, semantics.problemN,
                                        semantics.problemK}));
  attrs.set("tile_shape",
            buildI64ArrayAttr(builder, {semantics.tileM, semantics.tileN,
                                        semantics.tileK}));
  attrs.set("exact_tile", builder.getBoolAttr(semantics.exactTile));
  attrs.set("boundary_m", builder.getBoolAttr(semantics.hasBoundaryM));
  attrs.set("boundary_n", builder.getBoolAttr(semantics.hasBoundaryN));
  attrs.set("boundary_k", builder.getBoolAttr(semantics.hasBoundaryK));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<MatmulSemantics> mlir::tb::parseMatmulSemanticsAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.semantic_matmul"));
  if (!root) {
    op->emitError() << "missing `tb.semantic_matmul` attribute";
    return failure();
  }

  MatmulSemantics semantics;
  auto contractModel = readStringField(root, "contract_model", op);
  auto aDescType = dyn_cast_or_null<TypeAttr>(root.get("a_desc_type"));
  auto bDescType = dyn_cast_or_null<TypeAttr>(root.get("b_desc_type"));
  auto cDescType = dyn_cast_or_null<TypeAttr>(root.get("c_desc_type"));
  auto problemShape = dyn_cast_or_null<DenseI64ArrayAttr>(root.get("problem_shape"));
  auto tileShape = dyn_cast_or_null<DenseI64ArrayAttr>(root.get("tile_shape"));
  auto exactTile = readBoolField(root, "exact_tile", op);
  auto boundaryM = readBoolField(root, "boundary_m", op);
  auto boundaryN = readBoolField(root, "boundary_n", op);
  auto boundaryK = readBoolField(root, "boundary_k", op);
  if (failed(contractModel) || !aDescType || !bDescType || !cDescType ||
      !problemShape || !tileShape || failed(exactTile) || failed(boundaryM) ||
      failed(boundaryN) || failed(boundaryK)) {
    op->emitError() << "malformed `tb.semantic_matmul` attribute";
    return failure();
  }

  SmallVector<int64_t> parsedProblemShape = parseI64Array(problemShape);
  SmallVector<int64_t> parsedTileShape = parseI64Array(tileShape);
  if (parsedProblemShape.size() != 3 || parsedTileShape.size() != 3) {
    op->emitError() << "`tb.semantic_matmul` shape fields must have ranks 3/3";
    return failure();
  }
  semantics.contractModel = contractModel->str();
  semantics.aDescType = aDescType.getValue();
  semantics.bDescType = bDescType.getValue();
  semantics.cDescType = cDescType.getValue();
  semantics.problemM = parsedProblemShape[0];
  semantics.problemN = parsedProblemShape[1];
  semantics.problemK = parsedProblemShape[2];
  semantics.tileM = parsedTileShape[0];
  semantics.tileN = parsedTileShape[1];
  semantics.tileK = parsedTileShape[2];
  semantics.exactTile = *exactTile;
  semantics.hasBoundaryM = *boundaryM;
  semantics.hasBoundaryN = *boundaryN;
  semantics.hasBoundaryK = *boundaryK;
  if (failed(validateMatmulSemantics(semantics, op))) {
    op->emitError() << "malformed `tb.semantic_matmul` attribute";
    return failure();
  }
  return semantics;
}

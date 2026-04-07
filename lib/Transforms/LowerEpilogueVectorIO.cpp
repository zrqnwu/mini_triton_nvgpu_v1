#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::tb;

namespace mlir::tb {
#define GEN_PASS_DEF_TBLOWEREPILOGUEVECTORIO
#include "tb/Transforms/Passes.h.inc"
} // namespace mlir::tb

namespace {

static FailureOr<Value> computeGlobalByteAddress(OpBuilder &builder, Location loc,
                                                 Value memrefValue, Value row,
                                                 Value col, StringRef role) {
  auto memrefType = dyn_cast<MemRefType>(memrefValue.getType());
  if (!memrefType || memrefType.getRank() != 2) {
    emitError(loc) << role << " requires a rank-2 memref source";
    return failure();
  }
  Type elementType = memrefType.getElementType();
  if (!elementType.isIntOrFloat() ||
      elementType.getIntOrFloatBitWidth() != 32) {
    emitError(loc) << role << " currently requires 32-bit integer/float elements";
    return failure();
  }
  auto metadata = memref::ExtractStridedMetadataOp::create(builder, loc,
                                                           memrefValue);
  Value basePointer =
      memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                     metadata.getBaseBuffer());
  Value linearOffset = metadata.getOffset();
  linearOffset = arith::AddIOp::create(
      builder, loc, linearOffset,
      arith::MulIOp::create(builder, loc, row, metadata.getStrides()[0]));
  linearOffset = arith::AddIOp::create(
      builder, loc, linearOffset,
      arith::MulIOp::create(builder, loc, col, metadata.getStrides()[1]));
  Value byteOffset = arith::MulIOp::create(
      builder, loc, linearOffset,
      arith::ConstantIndexOp::create(builder, loc,
                                     elementType.getIntOrFloatBitWidth() / 8));
  Value byteAddress =
      arith::AddIOp::create(builder, loc, basePointer, byteOffset);
  return arith::IndexCastUIOp::create(builder, loc, builder.getI64Type(),
                                      byteAddress)
      .getResult();
}

static Value buildZeroVector(OpBuilder &builder, Location loc,
                             VectorType vectorType) {
  Type elementType = vectorType.getElementType();
  if (!elementType.isF32()) {
    emitError(loc) << "late epilogue vector IO currently expects f32 vectors";
    return Value();
  }
  return arith::ConstantOp::create(
      builder, loc, vectorType,
      DenseElementsAttr::get(vectorType, builder.getF32FloatAttr(0.0)));
}

static bool isSupportedVectorWidth(int64_t vectorWidth) {
  return vectorWidth == 2 || vectorWidth == 4;
}

static LogicalResult validateVectorIOShape(Operation *op, VectorType vectorType,
                                           int64_t vectorWidth,
                                           StringRef role) {
  if (!vectorType || vectorType.getRank() != 1 ||
      !isSupportedVectorWidth(vectorWidth) ||
      vectorType.getShape().front() != vectorWidth ||
      !vectorType.getElementType().isF32()) {
    op->emitError() << role
                    << " currently requires vector<2xf32> or vector<4xf32> "
                       "payloads";
    return failure();
  }
  return success();
}

static FailureOr<Value> getMemrefDimValue(OpBuilder &builder, Location loc,
                                          Value memrefValue, int64_t dim,
                                          StringRef role) {
  auto memrefType = dyn_cast<MemRefType>(memrefValue.getType());
  if (!memrefType || memrefType.getRank() != 2) {
    emitError(loc) << role << " requires a rank-2 memref operand";
    return failure();
  }
  if (!ShapedType::isDynamic(memrefType.getShape()[dim]))
    return arith::ConstantIndexOp::create(builder, loc, memrefType.getShape()[dim])
        .getResult();
  return memref::DimOp::create(builder, loc, memrefValue, dim).getResult();
}

static Value buildBoundaryMask(OpBuilder &builder, Location loc, Value row,
                               Value colBase, int64_t vectorWidth,
                               Value problemRows, Value problemCols) {
  VectorType maskType = VectorType::get({vectorWidth}, builder.getI1Type());
  Value mask = arith::ConstantOp::create(
      builder, loc, maskType,
      DenseElementsAttr::get(maskType, builder.getBoolAttr(false)));
  Value rowInBounds = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, row, problemRows);
  for (int64_t i = 0; i < vectorWidth; ++i) {
    Value laneCol = arith::AddIOp::create(
        builder, loc, colBase, arith::ConstantIndexOp::create(builder, loc, i));
    Value colInBounds = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ult, laneCol, problemCols);
    Value laneInBounds =
        arith::AndIOp::create(builder, loc, rowInBounds, colInBounds);
    mask = vector::InsertOp::create(builder, loc, laneInBounds, mask,
                                    ArrayRef<int64_t>{i});
  }
  return mask;
}

static Value buildWholeVectorInBoundsPredicate(OpBuilder &builder, Location loc,
                                               Value row, Value colBase,
                                               int64_t vectorWidth,
                                               Value problemRows,
                                               Value problemCols) {
  Value rowInBounds = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, row, problemRows);
  Value lastCol = arith::AddIOp::create(
      builder, loc, colBase,
      arith::ConstantIndexOp::create(builder, loc, vectorWidth - 1));
  Value colInBounds = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, lastCol, problemCols);
  return arith::AndIOp::create(builder, loc, rowInBounds, colInBounds);
}

static FailureOr<Value> emitInlinePtxVectorLoad(OpBuilder &builder, Location loc,
                                                Value address,
                                                VectorType vectorType,
                                                int64_t vectorWidth) {
  SmallVector<Type, 4> resultTypes(static_cast<size_t>(vectorWidth),
                                   builder.getF32Type());
  StringRef asmString = vectorWidth == 4
                            ? "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "ld.global.v2.b32 {%0, %1}, [%2];";
  auto inlinePtx = NVVM::InlinePtxOp::create(builder, loc, resultTypes,
                                             ValueRange{address}, ValueRange{},
                                             asmString, Value());
  Value vec = buildZeroVector(builder, loc, vectorType);
  if (!vec)
    return failure();
  for (int64_t i = 0; i < vectorWidth; ++i) {
    vec = vector::InsertOp::create(builder, loc, inlinePtx.getResult(i), vec,
                                   ArrayRef<int64_t>{i});
  }
  return vec;
}

static FailureOr<Value>
emitInlinePtxPredicatedVectorLoad(OpBuilder &builder, Location loc, Value address,
                                  Value predicateI32, VectorType vectorType,
                                  int64_t vectorWidth) {
  SmallVector<Type, 4> resultTypes(static_cast<size_t>(vectorWidth),
                                   builder.getF32Type());
  StringRef asmString =
      vectorWidth == 4
          ? "{ .reg .pred p; setp.ne.u32 p, %5, 0; mov.b32 %0, 0; mov.b32 %1, "
            "0; mov.b32 %2, 0; mov.b32 %3, 0; @p ld.global.v4.b32 {%0, %1, "
            "%2, %3}, [%4]; }"
          : "{ .reg .pred p; setp.ne.u32 p, %3, 0; mov.b32 %0, 0; mov.b32 %1, "
            "0; @p ld.global.v2.b32 {%0, %1}, [%2]; }";
  auto inlinePtx = NVVM::InlinePtxOp::create(
      builder, loc, resultTypes, ValueRange{address, predicateI32},
      ValueRange{}, asmString, Value());
  Value vec = buildZeroVector(builder, loc, vectorType);
  if (!vec)
    return failure();
  for (int64_t i = 0; i < vectorWidth; ++i) {
    vec = vector::InsertOp::create(builder, loc, inlinePtx.getResult(i), vec,
                                   ArrayRef<int64_t>{i});
  }
  return vec;
}

static void emitInlinePtxVectorStore(OpBuilder &builder, Location loc,
                                     Value address, Value vectorValue,
                                     int64_t vectorWidth) {
  SmallVector<Value, 5> args;
  args.push_back(address);
  for (int64_t i = 0; i < vectorWidth; ++i) {
    args.push_back(vector::ExtractOp::create(builder, loc, vectorValue,
                                             ArrayRef<int64_t>{i}));
  }
  StringRef asmString = vectorWidth == 4
                            ? "st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                            : "st.global.v2.b32 [%0], {%1, %2};";
  NVVM::InlinePtxOp::create(builder, loc, TypeRange{}, args, ValueRange{},
                            asmString, Value());
}

static void emitInlinePtxPredicatedVectorStore(OpBuilder &builder, Location loc,
                                               Value address,
                                               Value vectorValue,
                                               Value predicateI32,
                                               int64_t vectorWidth) {
  SmallVector<Value, 6> args;
  args.push_back(address);
  for (int64_t i = 0; i < vectorWidth; ++i) {
    args.push_back(vector::ExtractOp::create(builder, loc, vectorValue,
                                             ArrayRef<int64_t>{i}));
  }
  args.push_back(predicateI32);
  StringRef asmString =
      vectorWidth == 4
          ? "{ .reg .pred p; setp.ne.u32 p, %5, 0; @p st.global.v4.b32 [%0], "
            "{%1, %2, %3, %4}; }"
          : "{ .reg .pred p; setp.ne.u32 p, %3, 0; @p st.global.v2.b32 [%0], "
            "{%1, %2}; }";
  NVVM::InlinePtxOp::create(builder, loc, TypeRange{}, args, ValueRange{},
                            asmString, Value());
}

class TBLowerEpilogueVectorIO
    : public mlir::tb::impl::TBLowerEpilogueVectorIOBase<
          TBLowerEpilogueVectorIO> {
public:
  using mlir::tb::impl::TBLowerEpilogueVectorIOBase<
      TBLowerEpilogueVectorIO>::TBLowerEpilogueVectorIOBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    NVVM::NVVMDialect, vector::VectorDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool hadFailure = false;

    SmallVector<EpilogueGlobalVectorLoadOp, 32> loads;
    SmallVector<EpilogueGlobalVectorStoreOp, 32> stores;
    module.walk([&](EpilogueGlobalVectorLoadOp op) { loads.push_back(op); });
    module.walk([&](EpilogueGlobalVectorStoreOp op) { stores.push_back(op); });

    for (EpilogueGlobalVectorLoadOp op : loads) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto vectorType = dyn_cast<VectorType>(op.getResult().getType());
      if (failed(validateVectorIOShape(op, vectorType, op.getVectorWidth(),
                                       "epilogue load"))) {
        hadFailure = true;
        continue;
      }
      Value vec;
      if (!op.getBoundaryAware()) {
        auto address = computeGlobalByteAddress(builder, loc, op.getSource(),
                                                op.getRow(), op.getCol(),
                                                "epilogue load");
        if (failed(address)) {
          hadFailure = true;
          continue;
        }
        auto loaded = emitInlinePtxVectorLoad(builder, loc, *address, vectorType,
                                              op.getVectorWidth());
        if (failed(loaded)) {
          hadFailure = true;
          continue;
        }
        vec = *loaded;
      } else if (!op.getScalarTail()) {
        auto address = computeGlobalByteAddress(builder, loc, op.getSource(),
                                                op.getRow(), op.getCol(),
                                                "epilogue load");
        auto problemRows =
            getMemrefDimValue(builder, loc, op.getSource(), 0, "epilogue load");
        auto problemCols =
            getMemrefDimValue(builder, loc, op.getSource(), 1, "epilogue load");
        if (failed(address) || failed(problemRows) || failed(problemCols)) {
          hadFailure = true;
          continue;
        }
        Value wholeVectorInBounds = buildWholeVectorInBoundsPredicate(
            builder, loc, op.getRow(), op.getCol(), op.getVectorWidth(),
            *problemRows, *problemCols);
        Value predicateI32 = arith::ExtUIOp::create(
            builder, loc, builder.getI32Type(), wholeVectorInBounds);
        auto loaded = emitInlinePtxPredicatedVectorLoad(
            builder, loc, *address, predicateI32, vectorType,
            op.getVectorWidth());
        if (failed(loaded)) {
          hadFailure = true;
          continue;
        }
        vec = *loaded;
      } else {
        auto problemRows =
            getMemrefDimValue(builder, loc, op.getSource(), 0, "epilogue load");
        auto problemCols =
            getMemrefDimValue(builder, loc, op.getSource(), 1, "epilogue load");
        if (failed(problemRows) || failed(problemCols)) {
          hadFailure = true;
          continue;
        }
        Value passThru = buildZeroVector(builder, loc, vectorType);
        if (!passThru) {
          hadFailure = true;
          continue;
        }
        Value mask =
            buildBoundaryMask(builder, loc, op.getRow(), op.getCol(),
                              op.getVectorWidth(), *problemRows, *problemCols);
        vec = vector::MaskedLoadOp::create(builder, loc, vectorType,
                                           op.getSource(),
                                           ValueRange{op.getRow(), op.getCol()},
                                           mask, passThru)
                  .getResult();
      }
      op.replaceAllUsesWith(vec);
      op.erase();
    }

    for (EpilogueGlobalVectorStoreOp op : stores) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto vectorType = dyn_cast<VectorType>(op.getValue().getType());
      if (failed(validateVectorIOShape(op, vectorType, op.getVectorWidth(),
                                       "epilogue store"))) {
        hadFailure = true;
        continue;
      }
      if (!op.getBoundaryAware()) {
        auto address = computeGlobalByteAddress(builder, loc, op.getDest(),
                                                op.getRow(), op.getCol(),
                                                "epilogue store");
        if (failed(address)) {
          hadFailure = true;
          continue;
        }
        emitInlinePtxVectorStore(builder, loc, *address, op.getValue(),
                                 op.getVectorWidth());
      } else if (!op.getScalarTail()) {
        auto address = computeGlobalByteAddress(builder, loc, op.getDest(),
                                                op.getRow(), op.getCol(),
                                                "epilogue store");
        auto problemRows =
            getMemrefDimValue(builder, loc, op.getDest(), 0, "epilogue store");
        auto problemCols =
            getMemrefDimValue(builder, loc, op.getDest(), 1, "epilogue store");
        if (failed(address) || failed(problemRows) || failed(problemCols)) {
          hadFailure = true;
          continue;
        }
        Value wholeVectorInBounds = buildWholeVectorInBoundsPredicate(
            builder, loc, op.getRow(), op.getCol(), op.getVectorWidth(),
            *problemRows, *problemCols);
        Value predicateI32 = arith::ExtUIOp::create(
            builder, loc, builder.getI32Type(), wholeVectorInBounds);
        emitInlinePtxPredicatedVectorStore(builder, loc, *address, op.getValue(),
                                           predicateI32, op.getVectorWidth());
      } else {
        auto problemRows =
            getMemrefDimValue(builder, loc, op.getDest(), 0, "epilogue store");
        auto problemCols =
            getMemrefDimValue(builder, loc, op.getDest(), 1, "epilogue store");
        if (failed(problemRows) || failed(problemCols)) {
          hadFailure = true;
          continue;
        }
        Value mask =
            buildBoundaryMask(builder, loc, op.getRow(), op.getCol(),
                              op.getVectorWidth(), *problemRows, *problemCols);
        vector::MaskedStoreOp::create(builder, loc, op.getDest(),
                                      ValueRange{op.getRow(), op.getCol()},
                                      mask, op.getValue());
      }
      op.erase();
    }

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

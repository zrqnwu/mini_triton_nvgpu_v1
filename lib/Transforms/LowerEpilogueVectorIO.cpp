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

static LogicalResult validateVectorIOShape(Operation *op, VectorType vectorType,
                                           int64_t vectorWidth,
                                           StringRef role) {
  if (!vectorType || vectorType.getRank() != 1 || vectorWidth != 4 ||
      vectorType.getShape().front() != vectorWidth ||
      !vectorType.getElementType().isF32()) {
    op->emitError() << role << " currently requires vector<4xf32> payloads";
    return failure();
  }
  return success();
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
      if (op.getBoundaryAware()) {
        op.emitError() << "late epilogue vector IO currently requires "
                          "exact-tile non-boundary-aware loads";
        hadFailure = true;
        continue;
      }
      auto vectorType = dyn_cast<VectorType>(op.getResult().getType());
      if (failed(validateVectorIOShape(op, vectorType, op.getVectorWidth(),
                                       "epilogue load"))) {
        hadFailure = true;
        continue;
      }
      auto address = computeGlobalByteAddress(builder, loc, op.getSource(),
                                              op.getRow(), op.getCol(),
                                              "epilogue load");
      if (failed(address)) {
        hadFailure = true;
        continue;
      }
      auto inlinePtx = NVVM::InlinePtxOp::create(
          builder, loc,
          TypeRange{builder.getF32Type(), builder.getF32Type(),
                    builder.getF32Type(), builder.getF32Type()},
          ValueRange{*address}, ValueRange{},
          "ld.global.v4.b32 {%0, %1, %2, %3}, [%4];", Value());
      Value vec = buildZeroVector(builder, loc, vectorType);
      if (!vec) {
        hadFailure = true;
        continue;
      }
      for (int64_t i = 0; i < 4; ++i) {
        vec = vector::InsertOp::create(builder, loc, inlinePtx.getResult(i), vec,
                                       ArrayRef<int64_t>{i});
      }
      op.replaceAllUsesWith(vec);
      op.erase();
    }

    for (EpilogueGlobalVectorStoreOp op : stores) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      if (op.getBoundaryAware()) {
        op.emitError() << "late epilogue vector IO currently requires "
                          "exact-tile non-boundary-aware stores";
        hadFailure = true;
        continue;
      }
      auto vectorType = dyn_cast<VectorType>(op.getValue().getType());
      if (failed(validateVectorIOShape(op, vectorType, op.getVectorWidth(),
                                       "epilogue store"))) {
        hadFailure = true;
        continue;
      }
      auto address = computeGlobalByteAddress(builder, loc, op.getDest(),
                                              op.getRow(), op.getCol(),
                                              "epilogue store");
      if (failed(address)) {
        hadFailure = true;
        continue;
      }
      SmallVector<Value, 5> args;
      args.push_back(*address);
      for (int64_t i = 0; i < 4; ++i) {
        args.push_back(vector::ExtractOp::create(
            builder, loc, op.getValue(), ArrayRef<int64_t>{i}));
      }
      NVVM::InlinePtxOp::create(builder, loc, TypeRange{}, args, ValueRange{},
                                "st.global.v4.b32 [%0], {%1, %2, %3, %4};",
                                Value());
      op.erase();
    }

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

#include "tb/Analysis/KernelConfig.h"

#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"

#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/MatmulSemantics.h"
#include "tb/Analysis/PipelineReady.h"
#include "tb/Analysis/ProgramMappingPlan.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static FailureOr<ScalarKind> getScalarKind(Type type, Operation *op,
                                           StringRef name) {
  if (type.isF16())
    return ScalarKind::F16;
  if (type.isF32())
    return ScalarKind::F32;
  op->emitError() << "unsupported element type for `" << name
                  << "`: expected f16/f32";
  return failure();
}

static FailureOr<int64_t> readRequiredModuleI64Attr(Operation *op,
                                                    StringRef name) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) {
    op->emitError() << "kernel config requires a parent module";
    return failure();
  }
  auto attr = dyn_cast_or_null<IntegerAttr>(module->getAttr(name));
  if (!attr) {
    op->emitError() << "missing module context attr `" << name << "`";
    return failure();
  }
  return attr.getInt();
}

static FailureOr<MmaKind> getMmaKindFromEncodingPlan(const EncodingPlan &plan,
                                                     Operation *op) {
  auto accEncoding =
      getAccumulatorEncodingAttr(plan, plan.acc, op, "pipeline accumulator");
  if (failed(accEncoding))
    return failure();
  NVGPUMmaEncodingAttr mma = (*accEncoding).getParent();
  if (mma.getMmaFamily() == "mma_sync" && mma.getInstrShape().size() == 3 &&
      mma.getInstrShape()[0] == 16 && mma.getInstrShape()[1] == 8 &&
      mma.getInstrShape()[2] == 16) {
    return MmaKind::M16N8K16;
  }
  op->emitError() << "unsupported pipeline mma encoding owner truth";
  return failure();
}

} // namespace

StringRef mlir::tb::stringifyScalarKind(ScalarKind kind) {
  switch (kind) {
  case ScalarKind::F16:
    return "f16";
  case ScalarKind::F32:
    return "f32";
  }
  llvm_unreachable("unknown scalar kind");
}

StringRef mlir::tb::stringifyMmaKind(MmaKind kind) {
  switch (kind) {
  case MmaKind::M16N8K16:
    return "m16n8k16";
  }
  llvm_unreachable("unknown mma kind");
}

static FailureOr<KernelConfig> deriveSourceKernelConfig(MatmulOp op) {
  auto aType = dyn_cast<MemRefType>(op.getA().getType());
  auto bType = dyn_cast<MemRefType>(op.getB().getType());
  auto cType = dyn_cast<MemRefType>(op.getC().getType());
  if (!aType || !bType || !cType) {
    op.emitError() << "tb.matmul operands must be memrefs";
    return failure();
  }
  if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2) {
    op.emitError() << "tb.matmul currently requires rank-2 memrefs";
    return failure();
  }
  if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
      !cType.hasStaticShape()) {
    op.emitError() << "stage1 supported envelope currently requires static "
                      "rank-2 memrefs to derive problem shape coverage";
    return failure();
  }

  auto aScalar = getScalarKind(aType.getElementType(), op, "a");
  auto bScalar = getScalarKind(bType.getElementType(), op, "b");
  auto cScalar = getScalarKind(cType.getElementType(), op, "c");
  auto numWarps = readRequiredModuleI64Attr(op.getOperation(), kTBNumWarpsAttrName);
  auto requestedStages =
      readRequiredModuleI64Attr(op.getOperation(), kTBRequestedStagesAttrName);
  if (failed(aScalar) || failed(bScalar) || failed(cScalar) ||
      failed(numWarps) || failed(requestedStages)) {
    return failure();
  }

  KernelConfig config;
  config.problemM = aType.getShape()[0];
  config.problemK = aType.getShape()[1];
  config.problemN = bType.getShape()[1];
  config.blockM = op.getBlockM();
  config.blockN = op.getBlockN();
  config.blockK = op.getBlockK();
  config.numWarps = *numWarps;
  config.requestedStages = *requestedStages;
  config.groupM = op.getGroupM();
  config.exactTile = op.getExactTile();
  config.mmaKind = MmaKind::M16N8K16;
  config.aScalar = *aScalar;
  config.bScalar = *bScalar;
  config.cScalar = *cScalar;
  return config;
}

static FailureOr<KernelConfig> derivePipelineKernelConfig(PipelineMainlineOp op) {
  auto aType = dyn_cast<MemRefType>(op.getA().getType());
  auto bType = dyn_cast<MemRefType>(op.getB().getType());
  auto cType = dyn_cast<MemRefType>(op.getC().getType());
  if (!aType || !bType || !cType) {
    op.emitError() << "tb.pipeline_mainline operands must be memrefs";
    return failure();
  }
  if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2) {
    op.emitError() << "tb.pipeline_mainline currently requires rank-2 memrefs";
    return failure();
  }
  if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
      !cType.hasStaticShape()) {
    op.emitError() << "pipeline mainline requires static rank-2 memrefs";
    return failure();
  }

  auto semantics = parseMatmulSemanticsAttr(op.getOperation());
  auto mapping = parseProgramMappingPlanAttr(op.getOperation());
  auto encodings = parseEncodingPlanAttr(op.getOperation());
  auto ready = parsePipelineReadyAttr(op.getOperation());
  auto aScalar = getScalarKind(aType.getElementType(), op, "a");
  auto bScalar = getScalarKind(bType.getElementType(), op, "b");
  auto cScalar = getScalarKind(cType.getElementType(), op, "c");
  if (failed(semantics) || failed(mapping) || failed(encodings) ||
      failed(ready) || failed(aScalar) || failed(bScalar) || failed(cScalar)) {
    return failure();
  }

  auto numWarps = readRequiredModuleI64Attr(op.getOperation(), kTBNumWarpsAttrName);
  auto mmaKind = getMmaKindFromEncodingPlan(*encodings, op.getOperation());
  if (failed(numWarps) || failed(mmaKind))
    return failure();

  if (mapping->tileM != semantics->tileM || mapping->tileN != semantics->tileN ||
      mapping->tileK != semantics->tileK) {
    op.emitError() << "pipeline program mapping tile shape disagrees with "
                      "semantic tile ownership";
    return failure();
  }

  KernelConfig config;
  config.problemM = semantics->problemM;
  config.problemN = semantics->problemN;
  config.problemK = semantics->problemK;
  config.blockM = semantics->tileM;
  config.blockN = semantics->tileN;
  config.blockK = semantics->tileK;
  config.numWarps = *numWarps;
  config.requestedStages = ready->requestedStages;
  config.groupM = mapping->groupM;
  config.exactTile = semantics->exactTile;
  config.mmaKind = *mmaKind;
  config.aScalar = *aScalar;
  config.bScalar = *bScalar;
  config.cScalar = *cScalar;
  return config;
}

FailureOr<KernelConfig> mlir::tb::getKernelConfig(MatmulOp op) {
  return deriveSourceKernelConfig(op);
}

FailureOr<KernelConfig> mlir::tb::getKernelConfig(PipelineMainlineOp op) {
  return derivePipelineKernelConfig(op);
}

LogicalResult mlir::tb::verifySupportedKernelConfig(const KernelConfig &config,
                                                    Operation *op) {
  if (config.aScalar != ScalarKind::F16 || config.bScalar != ScalarKind::F16 ||
      config.cScalar != ScalarKind::F32) {
    return op->emitError()
           << "stage1 currently supports fp16 x fp16 -> fp32 matmul kernels";
  }
  if (config.mmaKind != MmaKind::M16N8K16)
    return op->emitError() << "stage1 currently supports mma = `m16n8k16`";
  if (config.problemM <= 0 || config.problemN <= 0 || config.problemK <= 0) {
    return op->emitError() << "problem M/N/K must be positive";
  }
  if (config.blockM <= 0 || config.blockN <= 0 || config.blockK <= 0) {
    return op->emitError() << "block M/N/K must be positive";
  }
  if (config.blockM % 16 != 0 || config.blockN % 8 != 0 || config.blockK % 16 != 0) {
    return op->emitError()
           << "stage1 requires block_m % 16 == 0, block_n % 8 == 0 and "
              "block_k % 16 == 0 for mma.sync m16n8k16 legality";
  }
  if (config.requestedStages < 2 || config.requestedStages > 4) {
    return op->emitError() << "stage1 currently supports num_stages in [2, 4]";
  }
  if (!(config.numWarps == 1 || config.numWarps == 4 || config.numWarps == 8)) {
    return op->emitError() << "stage1 currently supports num_warps in {1, 4, 8}";
  }
  if (config.groupM <= 0)
    return op->emitError() << "group_m must be positive";
  if (config.exactTile &&
      ((config.problemM % config.blockM) != 0 ||
       (config.problemN % config.blockN) != 0 ||
       (config.problemK % config.blockK) != 0)) {
    return op->emitError()
           << "exact_tile=true requires problem M/N/K to be divisible by "
              "block M/N/K";
  }
  if (!config.exactTile &&
      (ceilDiv(config.problemM, config.blockM) <= 0 ||
       ceilDiv(config.problemN, config.blockN) <= 0 ||
       ceilDiv(config.problemK, config.blockK) <= 0)) {
    return op->emitError() << "general-shape matmul must have at least one tile on "
                              "every problem axis";
  }
  return success();
}

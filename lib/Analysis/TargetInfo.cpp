#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"

#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

static ModuleOp getOwningModule(Operation *op) {
  return op ? op->getParentOfType<ModuleOp>() : ModuleOp();
}

static FailureOr<int64_t> readI64Field(DictionaryAttr dict, StringRef name,
                                       Operation *op) {
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing integer field `" << name << "`";
    return failure();
  }
  return attr.getInt();
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

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static FailureOr<DenseI64ArrayAttr>
readDenseI64ArrayField(DictionaryAttr dict, StringRef name, Operation *op) {
  auto attr = dyn_cast_or_null<DenseI64ArrayAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing dense i64 array field `" << name << "`";
    return failure();
  }
  return attr;
}

static SmallVector<int64_t, 4> parseI64Array(DenseI64ArrayAttr attr) {
  return SmallVector<int64_t, 4>(attr.asArrayRef().begin(),
                                 attr.asArrayRef().end());
}

static DenseI64ArrayAttr buildI64ArrayAttr(Builder &builder,
                                           ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

static FailureOr<DictionaryAttr> getTargetInfoOwnerAttr(Operation *op) {
  DictionaryAttr moduleRoot;
  if (auto module = getOwningModule(op)) {
    moduleRoot =
        dyn_cast_or_null<DictionaryAttr>(module->getAttr(kTBTargetModuleAttrName));
  }
  auto opRoot = dyn_cast_or_null<DictionaryAttr>(op->getAttr(kTBTargetOpAttrName));
  if (isa<PipelineMainlineOp>(op)) {
    if (opRoot) {
      op->emitError()
          << "strict pipeline_mainline must not retain legacy per-op target "
             "ownership `"
          << kTBTargetOpAttrName << "`";
      return failure();
    }
    if (!moduleRoot) {
      op->emitError() << "strict pipeline_mainline requires module target "
                         "ownership `"
                      << kTBTargetModuleAttrName << "`";
      return failure();
    }
    return moduleRoot;
  }
  if (moduleRoot && opRoot && moduleRoot != opRoot) {
    op->emitError() << "module target owner `" << kTBTargetModuleAttrName
                    << "` conflicts with legacy fallback `" << kTBTargetOpAttrName
                    << "`";
    return failure();
  }
  if (moduleRoot)
    return moduleRoot;
  if (opRoot)
    return opRoot;
  op->emitError() << "missing `" << kTBTargetModuleAttrName << "` or `"
                  << kTBTargetOpAttrName << "` attribute";
  return failure();
}

static FailureOr<int64_t> parseSmArch(StringRef gpuArch, Operation *op) {
  if (!gpuArch.consume_front("sm_") || gpuArch.empty()) {
    op->emitError() << "unsupported gpu arch `" << gpuArch
                    << "`; expected `sm_<major><minor>`";
    return failure();
  }
  int64_t sm = 0;
  if (gpuArch.getAsInteger(10, sm)) {
    op->emitError() << "gpu arch `" << gpuArch
                    << "` must carry a decimal SM version";
    return failure();
  }
  return sm;
}

static TargetInfo makeBaseTargetInfo(StringRef gpuArch, StringRef ptxFeatures) {
  TargetInfo target;
  target.gpuArch = gpuArch.str();
  target.targetTriple = "nvptx64-nvidia-cuda";
  target.ptxFeatures = ptxFeatures.str();
  target.threadsPerWarp = 32;
  target.sharedBankBytes = 4;
  target.numSms = 0;
  target.maxWarpsPerCTA = 8;
  target.maxRegistersPerThread = 255;
  target.maxRegistersPerCTA = 65536;
  target.maxSharedBytesPerCTA = 102400;
  target.supportsAsyncCopy = true;
  target.supportsLdMatrix = true;
  target.supportsMmaSync = true;
  target.supportsMBarrier = false;
  target.supportsTMA = false;
  target.supportsWGMMA = false;
  target.asyncCopyMinBytes = 4;
  target.asyncCopyMaxBytes = 16;
  target.asyncCopyPreferredBytes = 16;
  target.globalToSharedStageLatency = 2;
  target.sharedToRegisterStageLatency = 1;
  target.mmaStageLatency = 1;
  target.preferredAsyncTransport = "cp_async";
  target.mmaInstrShape = {16, 8, 16};
  return target;
}

static LogicalResult validateTargetInfo(const TargetInfo &target, Operation *op) {
  if (target.gpuArch.empty() || target.targetTriple.empty())
    return op->emitError() << "target info must carry concrete gpu/triple";
  if (target.threadsPerWarp <= 0 || target.maxWarpsPerCTA <= 0)
    return op->emitError()
           << "target info must carry positive warp scheduling limits";
  if (target.asyncCopyMinBytes <= 0 || target.asyncCopyMaxBytes <= 0 ||
      target.asyncCopyPreferredBytes <= 0 ||
      target.asyncCopyMinBytes > target.asyncCopyMaxBytes) {
    return op->emitError() << "target info async-copy byte range is invalid";
  }
  if (target.globalToSharedStageLatency < 0 ||
      target.sharedToRegisterStageLatency < 0 || target.mmaStageLatency < 0) {
    return op->emitError() << "target info stage latencies must be non-negative";
  }
  if (target.mmaInstrShape.size() != 3 || target.mmaInstrShape[0] <= 0 ||
      target.mmaInstrShape[1] <= 0 || target.mmaInstrShape[2] <= 0) {
    return op->emitError()
           << "target info must carry a positive rank-3 mma instruction shape";
  }
  return success();
}

} // namespace

FailureOr<TargetInfo> mlir::tb::deriveTargetInfoForArch(StringRef gpuArch,
                                                        StringRef ptxFeatures,
                                                        Operation *op) {
  auto sm = parseSmArch(gpuArch, op);
  if (failed(sm))
    return failure();

  int64_t major = *sm / 10;
  int64_t minor = *sm % 10;
  (void)minor;

  TargetInfo target = makeBaseTargetInfo(gpuArch, ptxFeatures);
  if (*sm == 80) {
    target.numSms = 108;
    target.maxSharedBytesPerCTA = 163840;
  } else if (*sm == 86) {
    target.numSms = 84;
    target.maxSharedBytesPerCTA = 102400;
  } else if (*sm == 89) {
    target.numSms = 128;
    target.maxSharedBytesPerCTA = 102400;
  } else if (*sm >= 90) {
    target.maxSharedBytesPerCTA = 227328;
    target.supportsMBarrier = true;
    target.supportsTMA = true;
    target.supportsWGMMA = true;
  } else if (*sm >= 80) {
    target.maxSharedBytesPerCTA = 102400;
  } else {
    op->emitError() << "stage1 strict pipeline requires NVIDIA arch >= sm_80, got `"
                    << gpuArch << "`";
    return failure();
  }

  if (major >= 9) {
    target.supportsMBarrier = true;
    target.supportsTMA = true;
    target.supportsWGMMA = true;
  }

  if (failed(validateTargetInfo(target, op)))
    return failure();
  return target;
}

DictionaryAttr mlir::tb::buildTargetInfoAttr(Builder &builder,
                                             const TargetInfo &target) {
  NamedAttrList attrs;
  attrs.set("gpu_arch", builder.getStringAttr(target.gpuArch));
  attrs.set("target_triple", builder.getStringAttr(target.targetTriple));
  attrs.set("ptx_features", builder.getStringAttr(target.ptxFeatures));
  attrs.set("threads_per_warp",
            builder.getI64IntegerAttr(target.threadsPerWarp));
  attrs.set("shared_bank_bytes",
            builder.getI64IntegerAttr(target.sharedBankBytes));
  attrs.set("num_sms", builder.getI64IntegerAttr(target.numSms));
  attrs.set("max_warps_per_cta",
            builder.getI64IntegerAttr(target.maxWarpsPerCTA));
  attrs.set("max_registers_per_thread",
            builder.getI64IntegerAttr(target.maxRegistersPerThread));
  attrs.set("max_registers_per_cta",
            builder.getI64IntegerAttr(target.maxRegistersPerCTA));
  attrs.set("max_shared_bytes_per_cta",
            builder.getI64IntegerAttr(target.maxSharedBytesPerCTA));
  attrs.set("supports_async_copy", builder.getBoolAttr(target.supportsAsyncCopy));
  attrs.set("supports_ldmatrix", builder.getBoolAttr(target.supportsLdMatrix));
  attrs.set("supports_mma_sync", builder.getBoolAttr(target.supportsMmaSync));
  attrs.set("supports_mbarrier", builder.getBoolAttr(target.supportsMBarrier));
  attrs.set("supports_tma", builder.getBoolAttr(target.supportsTMA));
  attrs.set("supports_wgmma", builder.getBoolAttr(target.supportsWGMMA));
  attrs.set("async_copy_min_bytes",
            builder.getI64IntegerAttr(target.asyncCopyMinBytes));
  attrs.set("async_copy_max_bytes",
            builder.getI64IntegerAttr(target.asyncCopyMaxBytes));
  attrs.set("async_copy_preferred_bytes",
            builder.getI64IntegerAttr(target.asyncCopyPreferredBytes));
  attrs.set("global_to_shared_stage_latency",
            builder.getI64IntegerAttr(target.globalToSharedStageLatency));
  attrs.set("shared_to_register_stage_latency",
            builder.getI64IntegerAttr(target.sharedToRegisterStageLatency));
  attrs.set("mma_stage_latency",
            builder.getI64IntegerAttr(target.mmaStageLatency));
  attrs.set("preferred_async_transport",
            builder.getStringAttr(target.preferredAsyncTransport));
  attrs.set("mma_instr_shape", buildI64ArrayAttr(builder, target.mmaInstrShape));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<TargetInfo> mlir::tb::parseTargetInfoAttr(Operation *op) {
  auto root = getTargetInfoOwnerAttr(op);
  if (failed(root))
    return failure();

  auto gpuArch = readStringField(*root, "gpu_arch", op);
  auto targetTriple = readStringField(*root, "target_triple", op);
  auto ptxFeatures = readStringField(*root, "ptx_features", op);
  auto threadsPerWarp = readI64Field(*root, "threads_per_warp", op);
  auto sharedBankBytes = readI64Field(*root, "shared_bank_bytes", op);
  auto numSms = readI64Field(*root, "num_sms", op);
  auto maxWarpsPerCTA = readI64Field(*root, "max_warps_per_cta", op);
  auto maxRegistersPerThread =
      readI64Field(*root, "max_registers_per_thread", op);
  auto maxRegistersPerCTA = readI64Field(*root, "max_registers_per_cta", op);
  auto maxSharedBytesPerCTA =
      readI64Field(*root, "max_shared_bytes_per_cta", op);
  auto supportsAsyncCopy = readBoolField(*root, "supports_async_copy", op);
  auto supportsLdMatrix = readBoolField(*root, "supports_ldmatrix", op);
  auto supportsMmaSync = readBoolField(*root, "supports_mma_sync", op);
  auto supportsMBarrier = readBoolField(*root, "supports_mbarrier", op);
  auto supportsTMA = readBoolField(*root, "supports_tma", op);
  auto supportsWGMMA = readBoolField(*root, "supports_wgmma", op);
  auto asyncCopyMinBytes = readI64Field(*root, "async_copy_min_bytes", op);
  auto asyncCopyMaxBytes = readI64Field(*root, "async_copy_max_bytes", op);
  auto asyncCopyPreferredBytes =
      readI64Field(*root, "async_copy_preferred_bytes", op);
  auto globalToSharedStageLatency =
      readI64Field(*root, "global_to_shared_stage_latency", op);
  auto sharedToRegisterStageLatency =
      readI64Field(*root, "shared_to_register_stage_latency", op);
  auto mmaStageLatency = readI64Field(*root, "mma_stage_latency", op);
  auto preferredAsyncTransport =
      readStringField(*root, "preferred_async_transport", op);
  auto mmaInstrShape = readDenseI64ArrayField(*root, "mma_instr_shape", op);
  if (failed(gpuArch) || failed(targetTriple) || failed(ptxFeatures) ||
      failed(threadsPerWarp) || failed(sharedBankBytes) || failed(numSms) ||
      failed(maxWarpsPerCTA) || failed(maxRegistersPerThread) ||
      failed(maxRegistersPerCTA) || failed(maxSharedBytesPerCTA) ||
      failed(supportsAsyncCopy) || failed(supportsLdMatrix) ||
      failed(supportsMmaSync) || failed(supportsMBarrier) ||
      failed(supportsTMA) || failed(supportsWGMMA) ||
      failed(asyncCopyMinBytes) || failed(asyncCopyMaxBytes) ||
      failed(asyncCopyPreferredBytes) ||
      failed(globalToSharedStageLatency) ||
      failed(sharedToRegisterStageLatency) || failed(mmaStageLatency) ||
      failed(preferredAsyncTransport) || failed(mmaInstrShape)) {
    op->emitError() << "malformed target owner attr (`" << kTBTargetModuleAttrName
                    << "` / `" << kTBTargetOpAttrName << "`)";
    return failure();
  }

  TargetInfo target;
  target.gpuArch = gpuArch->str();
  target.targetTriple = targetTriple->str();
  target.ptxFeatures = ptxFeatures->str();
  target.threadsPerWarp = *threadsPerWarp;
  target.sharedBankBytes = *sharedBankBytes;
  target.numSms = *numSms;
  target.maxWarpsPerCTA = *maxWarpsPerCTA;
  target.maxRegistersPerThread = *maxRegistersPerThread;
  target.maxRegistersPerCTA = *maxRegistersPerCTA;
  target.maxSharedBytesPerCTA = *maxSharedBytesPerCTA;
  target.supportsAsyncCopy = *supportsAsyncCopy;
  target.supportsLdMatrix = *supportsLdMatrix;
  target.supportsMmaSync = *supportsMmaSync;
  target.supportsMBarrier = *supportsMBarrier;
  target.supportsTMA = *supportsTMA;
  target.supportsWGMMA = *supportsWGMMA;
  target.asyncCopyMinBytes = *asyncCopyMinBytes;
  target.asyncCopyMaxBytes = *asyncCopyMaxBytes;
  target.asyncCopyPreferredBytes = *asyncCopyPreferredBytes;
  target.globalToSharedStageLatency = *globalToSharedStageLatency;
  target.sharedToRegisterStageLatency = *sharedToRegisterStageLatency;
  target.mmaStageLatency = *mmaStageLatency;
  target.preferredAsyncTransport = preferredAsyncTransport->str();
  target.mmaInstrShape = parseI64Array(*mmaInstrShape);
  if (failed(validateTargetInfo(target, op))) {
    op->emitError() << "malformed target owner attr (`" << kTBTargetModuleAttrName
                    << "` / `" << kTBTargetOpAttrName << "`)";
    return failure();
  }
  return target;
}

FailureOr<TargetInfo> mlir::tb::getTargetInfo(Operation *op) {
  return parseTargetInfoAttr(op);
}

LogicalResult mlir::tb::setModuleContextAttr(ModuleOp module, StringRef name,
                                             Attribute value,
                                             Operation *anchorOp) {
  if (!module)
    return anchorOp->emitError() << "missing owning module for context attr `"
                                 << name << "`";
  if (Attribute existing = module->getAttr(name)) {
    if (existing != value) {
      return anchorOp->emitError()
             << "module context attr `" << name
             << "` conflicts with another kernel in the same module";
    }
    return success();
  }
  module->setAttr(name, value);
  return success();
}

#include "tb/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

static void buildTBStage1MainlinePipeline(
    OpPassManager &pm, const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  mlir::tb::TBAttachTargetInfoOptions targetOptions;
  targetOptions.gpuArch =
      options.cubinChip.empty() ? std::string() : options.cubinChip.getValue();
  targetOptions.ptxFeatures = options.cubinFeatures.empty()
                                  ? std::string()
                                  : options.cubinFeatures.getValue();

  pm.addPass(mlir::tb::createTBVerifyScope());
  pm.addPass(mlir::tb::createTBAttachTargetInfo(targetOptions));
  pm.addPass(mlir::tb::createTBSemanticizeMatmul());
  pm.addPass(mlir::tb::createTBBuildProgramMappingPlan());
  pm.addPass(mlir::tb::createTBBuildLayoutPlan());
  pm.addPass(mlir::tb::createTBBuildTransportPlan());
  pm.addPass(mlir::tb::createTBBuildCRegisterPlan());
  pm.addPass(mlir::tb::createTBRewriteMatmulMainloop());
  pm.addPass(mlir::tb::createTBCleanupLayoutConversions());
  pm.addPass(mlir::tb::createTBBuildMainloopGraph());
  pm.addPass(mlir::tb::createTBRegularizeKLoop());
  pm.addPass(mlir::tb::createTBAssignLatencies());
  pm.addPass(mlir::tb::createTBScheduleLoops());
  pm.addPass(mlir::tb::createTBDeriveWaits());
  pm.addPass(mlir::tb::createTBExpandPipeline());
  pm.addPass(mlir::tb::createTBCleanupPipeline());
  pm.addPass(mlir::tb::createTBLowerPipelineToNVGPU());
}

static void buildTBStage1NVVMSinkPipeline(
    OpPassManager &pm, const mlir::gpu::GPUToNVVMPipelineOptions &options) {
  pm.addPass(mlir::tb::createTBLowerEpilogueVectorIO());
  pm.addPass(createConvertNVGPUToNVVMPass());
  pm.addPass(createGpuKernelOutliningPass());
  // 中文标记：direct-global C pack 需要保留 vector load/store 到 LLVM/NVVM。
  // 这里不能再先走 vector-to-scf，否则会把已经对齐的 v4 global path
  // 提前 scalarize 成大量标量 load/store。
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addPass(createConvertNVVMToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(memref::createExpandStridedMetadataPass());

  GpuNVVMAttachTargetOptions nvvmTargetOptions;
  nvvmTargetOptions.triple = options.cubinTriple;
  nvvmTargetOptions.chip = options.cubinChip;
  nvvmTargetOptions.features = options.cubinFeatures;
  nvvmTargetOptions.optLevel = options.optLevel;
  nvvmTargetOptions.cmdOptions = options.cmdOptions;
  pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));
  pm.addPass(createLowerAffinePass());
  pm.addPass(createArithToLLVMConversionPass());
  ConvertIndexToLLVMPassOptions convertIndexToLLVMPassOpt;
  convertIndexToLLVMPassOpt.indexBitwidth = options.indexBitWidth;
  pm.addPass(createConvertIndexToLLVMPass(convertIndexToLLVMPassOpt));
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  ConvertGpuOpsToNVVMOpsOptions gpuToNvvmOptions;
  gpuToNvvmOptions.useBarePtrCallConv = options.kernelUseBarePtrCallConv;
  gpuToNvvmOptions.indexBitwidth = options.indexBitWidth;
  gpuToNvvmOptions.allowPatternRollback = options.allowPatternRollback;
  pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps(
      gpuToNvvmOptions));
  pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());

  GpuToLLVMConversionPassOptions gpuToLlvmOptions;
  gpuToLlvmOptions.hostBarePtrCallConv = options.hostUseBarePtrCallConv;
  gpuToLlvmOptions.kernelBarePtrCallConv = options.kernelUseBarePtrCallConv;
  pm.addPass(createGpuToLLVMConversionPass(gpuToLlvmOptions));

  GpuModuleToBinaryPassOptions gpuModuleToBinaryPassOptions;
  gpuModuleToBinaryPassOptions.compilationTarget = options.cubinFormat;
  pm.addPass(createGpuModuleToBinaryPass(gpuModuleToBinaryPassOptions));
  pm.addPass(createConvertMathToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

namespace mlir::tb {

void registerStage1PipelineBuilders() {
  PassPipelineRegistration<mlir::gpu::GPUToNVVMPipelineOptions>(
      "tb-stage1-full-to-nvvm-pipeline",
      "Run the strict stage1 matmul pipeline from tb.matmul to NVVM/PTX-ready "
      "GPU IR with the repo-native final lowering order.",
      [](OpPassManager &pm,
         const mlir::gpu::GPUToNVVMPipelineOptions &options) {
        buildTBStage1MainlinePipeline(pm, options);
        buildTBStage1NVVMSinkPipeline(pm, options);
      });

  PassPipelineRegistration<mlir::gpu::GPUToNVVMPipelineOptions>(
      "tb-stage1-nvgpu-to-nvvm-pipeline",
      "Run the repo-native NVGPU/NVVM/PTX sink pipeline on tb-lowered "
      "stage1 gpu.module IR.",
      [](OpPassManager &pm,
         const mlir::gpu::GPUToNVVMPipelineOptions &options) {
        buildTBStage1NVVMSinkPipeline(pm, options);
      });
}

} // namespace mlir::tb

#include "tb/IR/TBDialect.h"
#include "tb/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tb::registerPasses();
  mlir::tb::registerStage1PipelineBuilders();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<mlir::tb::TBDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Mini Triton V1 optimizer driver\n",
                        registry));
}

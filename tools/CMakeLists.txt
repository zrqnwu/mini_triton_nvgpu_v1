set(LIBS
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
  MLIRMiniTritonTB
)

add_llvm_executable(tb-opt tb-opt.cpp)
llvm_update_compile_flags(tb-opt)
target_link_libraries(tb-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(tb-opt)

import os

import lit.formats

config.name = "MINI_TRITON_TB"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.tb_obj_root, "test")

config.substitutions.append(("%tb-opt", os.path.join(config.tb_tools_dir, "tb-opt")))
config.substitutions.append(
    ("FileCheck", os.path.join(config.llvm_tools_dir, "FileCheck"))
)

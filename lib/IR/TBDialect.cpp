#include "tb/IR/TBDialect.h"
#include "tb/IR/TBAttrs.h"
#include "tb/IR/TBOps.h"
#include "tb/IR/TBTypes.h"

using namespace mlir;
using namespace mlir::tb;

#include "tb/IR/TBOpsDialect.cpp.inc"

void TBDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "tb/IR/TBOps.cpp.inc"
      >();
}

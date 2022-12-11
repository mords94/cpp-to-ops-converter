//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Lib/helpers.h"

#include <string>
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<kernel mlir file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<std::string> functionName("function", cl::desc("Function name"),
                                         cl::value_desc("string"),
                                         cl::Required);

void printDepthValue(mlir::Value value, int depth) {
  llvm::errs() << std::string(depth, '--') << value << " (" << depth << ")"
               << "\n";
}

mlir::Value getBlockArgumentValue(mlir::BlockArgument arg, int depth) {
  auto blockArgIndex = arg.getArgNumber();

  auto parentWhileBeforeArgs =
      cast<mlir::scf::WhileOp>(arg.getParentRegion()->getParentOp())
          .getConditionOp()
          .getArgs();

  for (auto a : llvm::enumerate(parentWhileBeforeArgs)) {
    if (a.index() == blockArgIndex) {

      // check if the block argument is one of the init values of the
      // scf.while
      if (a.value().isa<mlir::BlockArgument>()) {
        auto argumentYieldedByCondition =
            a.value().dyn_cast<mlir::BlockArgument>();

        auto conditionArgIndex = argumentYieldedByCondition.getArgNumber();

        mlir::Value bound =
            argumentYieldedByCondition.getOwner()->getParentOp()->getOperand(
                conditionArgIndex);

        return bound;
      }

      // check if the block argument is a global memref value
      if (auto loadOp =
              dyn_cast<mlir::memref::LoadOp>(a.value().getDefiningOp())) {

        return loadOp.getMemRef();
      }

      llvm::errs() << std::string(depth, '--')
                   << "Not a block argument:" << a.value() << "\n";
    }
  }
}

template <typename T> bool has_use(mlir::Value value) {
  return llvm::any_of(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

template <typename T> auto get_use_range(mlir::Value value) {
  return llvm::make_filter_range(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

template <typename T> auto get_use(mlir::Value value) {
  llvm::SmallVector<T> uses;

  for (auto &op : value.getUses()) {
    if (isa<T>(op.getOwner())) {
      uses.push_back(cast<T>(op.getOwner()));
    }
  }

  return uses;
};

void printTree(mlir::Value value, int depth = 0) {

  if (depth > 5) {
    return;
  }

  if (auto op = value.getDefiningOp()) {
    if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(op)) {
      printTree(loadOp.getMemRef(), depth + 1);
    } else {
      llvm::errs() << std::string(depth, '--') << value << " (" << depth << ")"
                   << "\n";
      for (auto operand : value.getDefiningOp()->getOperands()) {
        printTree(operand, depth + 1);
      }
    }
  } else if (value.isa<mlir::BlockArgument>()) {

    auto operand =
        getBlockArgumentValue(value.cast<mlir::BlockArgument>(), depth);
    printTree(operand, depth + 1);
  }
};

int main(int argc, char **argv) {

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ops compiler\n");

  mlir::DialectRegistry registry;
  registerAllDialects(registry);

  mlir::MLIRContext context;

  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();

  context.allowUnregisteredDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module;

  parseMlir(context, module, inputFilename.getValue());

  mlir::func::FuncOp funcOp =
      module->lookupSymbol<mlir::func::FuncOp>(functionName.getValue());

  module->walk([&](mlir::func::FuncOp f) {
    if (!f.getName().contains(functionName.getValue())) {
      f.erase();
    }

    return mlir::WalkResult::advance();
  });

  llvm::outs() << "Function: " << funcOp.getName() << "\n";

  llvm::json::Object function;
  function["name"] = llvm::json::Value(functionName.getValue());
  llvm::json::Array args;
  for (auto argEntry : llvm::enumerate(funcOp.getArguments())) {

    llvm::json::Object argObject;

    argObject["name"] =
        llvm::json::Value("arg" + std::to_string(argEntry.index()));

    bool isStore = has_use<mlir::memref::StoreOp>(argEntry.value());
    bool isLoad = has_use<mlir::memref::LoadOp>(argEntry.value());

    if (isStore) {

      auto argStoreUses = get_use<mlir::memref::StoreOp>(argEntry.value());
      // assert(argStoreUses.size() == 1 && "Argument is used by multiple
      // stores");

      auto storeOp = argStoreUses[0];

      // for all index ops (we use flat 1d index for 2D and 3D arrays,
      // therefore it will be only one)
      for (auto index : storeOp.getIndices()) {
        printTree(index.getDefiningOp()->getOperand(0), 0);
      }
    }

    argObject["read"] = llvm::json::Value(isLoad);
    argObject["write"] = llvm::json::Value(isStore);
    args.push_back(llvm::json::Value(std::move(argObject)));
  }

  function["args"] = llvm::json::Value(std::move(args));

  llvm::outs() << llvm::json::Value(std::move(function)) << "\n";

  return 0;
}

// %13 = arith.addi %9, %12 : i32 (0)
// -%9 = arith.addi %arg5, %8 : i32 (1)
// ---%c0_i32 = arith.constant 0 : i32 (3)
// --%8 = arith.muli %arg3, %arg4 : i32 (2)
// ----%c0_i32 = arith.constant 0 : i32 (4)
// ----%2 = memref.get_global @im : memref<1xi32> (4)
// -%12 = arith.muli %10, %11 : i32 (1)
// --%10 = arith.muli %arg2, %arg4 : i32 (2)
// ----%c0_i32 = arith.constant 0 : i32 (4)
// ----%2 = memref.get_global @im : memref<1xi32> (4)
// ---%1 = memref.get_global @jm : memref<1xi32> (3)
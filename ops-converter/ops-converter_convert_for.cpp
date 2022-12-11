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

#include "Lib/ExpressionTree.h"
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

static cl::opt<bool> enableDebug("opconv-debug", cl::desc("Enable debug"),
                                 cl::init(false));

void setLoopAttribute(mlir::Value &value, int parentWhileDepth = -1) {
  auto context = value.getDefiningOp()->getContext();

  if (parentWhileDepth > -1) {
    value.getDefiningOp()->setAttr(
        "loop", mlir::IntegerAttr::get(mlir::IntegerType::get(context, 8),
                                       parentWhileDepth));
  }
}

void setInductionAttribute(mlir::Value &value) {
  auto context = value.getDefiningOp()->getContext();

  value.getDefiningOp()->setAttr("induction", mlir::UnitAttr::get(context));
}

/**
 * @brief Traverse the MLIR tree and finds the initial value of the block
 * arguments
 */
mlir::Value getBlockArgumentValue(mlir::BlockArgument arg, int depth) {
  auto blockArgIndex = arg.getArgNumber();

  auto parentWhileOp =
      cast<mlir::scf::WhileOp>(arg.getParentRegion()->getParentOp());

  auto parentWhileDepth =
      parentWhileOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt();

  auto parentWhileBeforeArgs = parentWhileOp.getConditionOp().getArgs();

  // FIXME: Rewrite this completely because it is not correct
  //  If there are two loops with the same init value, then this will not work
  //  (the same loop index will be used for both loops)
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

        setLoopAttribute(bound, parentWhileDepth);
        setInductionAttribute(bound);

        return bound;
      }

      // check if the block argument is a global memref value
      // in some case a global memref values is route through the scf before
      // region
      if (auto loadOp =
              dyn_cast<mlir::memref::LoadOp>(a.value().getDefiningOp())) {

        auto memref = loadOp.getMemRef();
        setLoopAttribute(memref, parentWhileDepth);

        return memref;
      }

      llvm::errs() << std::string(depth, '  ')
                   << "Not a block argument:" << a.value() << "\n";
    }
  }
}

/**
 * @brief Check if value has a use of type T
 * @param value The value to check
 */
template <typename T> bool has_use(mlir::Value value) {
  return llvm::any_of(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

/**
 * @brief Returns an iterator for all uses of the value that are of type T
 * @param value The value to check
 */
template <typename T> auto get_use_range(mlir::Value value) {
  return llvm::make_filter_range(value.getUses(), [&](mlir::OpOperand &op) {
    return isa<T>(op.getOwner());
  });
};

/**
 * @brief Returns all uses of the value that are of type T
 * @param value The value to check
 */
template <typename T> llvm::SmallVector<T> get_use(mlir::Value value) {
  llvm::SmallVector<T> uses;

  for (auto &op : value.getUses()) {
    if (isa<T>(op.getOwner())) {
      uses.push_back(cast<T>(op.getOwner()));
    }
  }

  return uses;
};

int toInt(llvm::StringRef str) {
  int value;
  str.getAsInteger(10, value);
  return value;
}

bool isFunctionArgument(mlir::Value value) {
  if (value.isa<mlir::BlockArgument>()) {
    auto blockArg = value.dyn_cast<mlir::BlockArgument>();
    return isa<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
  }
  return false;
}

/**
 * @brief Recursively builds an Expression tree for the given value
 * @param node Root node of the expression tree
 * @param value The value to build the tree for
 * @param depth The depth of the value in the tree
 *
 */
void buildTree(Node &node, mlir::Value value, int depth = 0) {
  node.setDepth(depth);

  if (auto op = value.getDefiningOp()) {
    if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(op)) {
      // global memref
      node.setValue(loadOp.getMemRef());
    } else if (auto constantIntOp = dyn_cast<mlir::arith::ConstantIntOp>(op)) {
      // index offset
      node.setValue(constantIntOp.getResult());
    } else {
      // binary (addi or muli) operation
      Node *children[2] = {new Node(), new Node()};

      node.setLeft(children[0]);
      node.setRight(children[1]);
      node.setValue(value);
      for (auto en : llvm::enumerate(value.getDefiningOp()->getOperands())) {
        buildTree(*children[en.index()], en.value(), depth + 1);
      }
    }
  } else if (value.isa<mlir::BlockArgument>()) {
    // induction variable
    auto operand =
        getBlockArgumentValue(value.cast<mlir::BlockArgument>(), depth);

    node.setValue(operand);
  }
};

/**
 * @brief Counts the maximum depth of nested loops within the given loop
 * operation
 * @param loopOp The loop operation to check
 * @param depth The current depth
 */
uint8_t countLoopNestDepth(mlir::scf::WhileOp &outerLoop, uint8_t depth = 1) {
  auto max = depth;
  for (auto innerLoop :
       outerLoop.getAfter().getBlocks().front().getOps<mlir::scf::WhileOp>()) {
    uint8_t nextDepth = countLoopNestDepth(innerLoop, depth + 1);

    if (nextDepth > max) {
      max = nextDepth;
    }
  }

  return max;
}

/**
 * @brief Returns innermost loops within the given loop operation
 * @param loopOp The loop operation to check
 * @param loops The vector to store the innermost loops
 * @param depth The current depth
 * @param level The level of the innermost loops to return
 */
void getLoopsOnLevelRecursive(mlir::scf::WhileOp &loop,
                              mlir::SmallVector<mlir::scf::WhileOp> &loops,
                              uint8_t level = 1, uint8_t depth = 1) {
  if (depth == level) {
    loops.push_back(loop);
  }

  for (auto innerLoop :
       loop.getAfter().getBlocks().front().getOps<mlir::scf::WhileOp>()) {
    getLoopsOnLevelRecursive(innerLoop, loops, level, depth + 1);
  }
}

mlir::SmallVector<mlir::scf::WhileOp>
getLoopsOnLevel(mlir::scf::WhileOp &loop, uint8_t level, uint8_t depth = 0) {
  mlir::SmallVector<mlir::scf::WhileOp> loops;

  getLoopsOnLevelRecursive(loop, loops, level, depth);

  return loops;
}

llvm::StringMap<mlir::BlockArgument>
getFuncArgumentsUsedInLoop(mlir::scf::WhileOp &loop) {
  llvm::StringMap<mlir::BlockArgument> argumentMap;

  loop.walk([&](mlir::Operation *op) {
    if (isa<mlir::memref::LoadOp>(op) || isa<mlir::memref::StoreOp>(op)) {
      mlir::Value memref = op->getOperand(0);

      if (memref.isa<mlir::BlockArgument>()) {
        mlir::BlockArgument arg = memref.cast<mlir::BlockArgument>();
        argumentMap[Twine(arg.getArgNumber()).str()] = arg;
      }
    } else {

      const bool hasFunctionArgumentAsOperand =
          llvm::any_of(op->getOperands(), [&](mlir::Value value) {
            return isFunctionArgument(value);
          });

      if (hasFunctionArgumentAsOperand) {
        op->emitError("Argument is used by an unsupported operation " +
                      op->getName().getStringRef());
      }
    }
  });

  return argumentMap;
}

template <typename T>
void setReadOrStoreValues(mlir::BlockArgument arg,
                          llvm::json::Object &jsonContainer,
                          llvm::StringRef key) {

  if (has_use<T>(arg)) {
    auto argStoreUses = get_use<T>(arg);
    // we use flat 1d index for 2D and 3D arrays therefore it will be  only
    // one index
    auto index = argStoreUses[0].getIndices()[0];

    Node *tree = new Node();
    buildTree(*tree, index.getDefiningOp()->getOperand(0), 0);

    if (enableDebug) {
      tree->dump();
      argStoreUses[0].dump();
    }

    if (enableDebug) {
      jsonContainer["debug"] = tree->toJSON();
    }
    jsonContainer["dims"] = llvm::json::Value(tree->getMemrefSize());
    jsonContainer["stencil"] = llvm::json::Value(tree->getStencil());
    jsonContainer[key] = llvm::json::Value(true);

  } else {
    jsonContainer[key] = llvm::json::Value(false);
  }
}

llvm::json::Object getArgumentMetadata(mlir::BlockArgument arg, int index) {

  llvm::json::Object argObject;

  argObject["arg_index"] = llvm::json::Value(index);
  argObject["name"] =
      llvm::json::Value("arg" + Twine(arg.getArgNumber()).str());

  setReadOrStoreValues<mlir::memref::StoreOp>(arg, argObject, "is_write");
  setReadOrStoreValues<mlir::memref::LoadOp>(arg, argObject, "is_read");

  return argObject;
}

mlir::WalkResult setLoopDepthAttribute(mlir::scf::WhileOp &loop,
                                       uint8_t depth) {
  loop->setAttr("depth",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(loop.getContext(), 8), depth));

  auto initialValue = loop.getInits()[0];

  if (auto constantIntOp =
          dyn_cast<mlir::arith::ConstantIntOp>(initialValue.getDefiningOp())) {
    loop->setAttr("lb", constantIntOp.getValueAttr());
  }

  auto ubValue = cast<mlir::arith::CmpIOp>(
                     loop.getConditionOp().getOperand(0).getDefiningOp())
                     ->getOperand(1)
                     .getDefiningOp()
                     ->getOperand(0);

  if (auto ubGlobal =
          dyn_cast<mlir::memref::GetGlobalOp>(ubValue.getDefiningOp())) {

    loop->setAttr("ub", ubGlobal.getNameAttr());
  }

  loop.getAfter().walk([&](mlir::scf::WhileOp innerLoop) {
    return setLoopDepthAttribute(innerLoop, depth + 1);
  });

  return mlir::WalkResult::advance();
}

void collectLoopBounds(mlir::scf::WhileOp loop, llvm::json::Array &bounds) {

  auto lb = llvm::json::Value(
      Twine(loop->getAttr("lb").cast<mlir::IntegerAttr>().getInt()).str());
  auto ub = llvm::json::Value(
      loop->getAttr("ub").cast<mlir::FlatSymbolRefAttr>().getValue());
  bounds.insert(bounds.begin(), ub);
  bounds.insert(bounds.begin(), lb);

  if (loop->getParentOfType<mlir::scf::WhileOp>()) {
    collectLoopBounds(loop->getParentOfType<mlir::scf::WhileOp>(), bounds);
  }
}

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

  llvm::json::Object function;
  function["name"] = llvm::json::Value(functionName.getValue());
  u_int16_t loopNestCounter = 0;
  llvm::json::Array loops;

  funcOp.walk(
      [&](mlir::scf::WhileOp loop) { return setLoopDepthAttribute(loop, 1); });

  funcOp.walk([&](mlir::scf::WhileOp loop) {
    auto beforeBuilder =
        mlir::OpBuilder::atBlockBegin(&loop->getParentRegion()->front());

    auto builder =
        mlir::OpBuilder::atBlockTerminator(&loop->getParentRegion()->front());

    auto rewriter = mlir::ConversionPatternRewriter(loop->getContext());

    auto zeroIndexOp =
        builder.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);

    auto lb = loop.getInits()[0];

    // TODO: handle constants too
    auto ub = dyn_cast<mlir::arith::CmpIOp>(
                  loop.getConditionOp().getOperand(0).getDefiningOp())
                  ->getOperand(1)
                  .getDefiningOp()
                  ->getOperand(0);

    auto ubLoad = builder.create<mlir::memref::LoadOp>(
        loop.getLoc(), ub, mlir::ValueRange({zeroIndexOp}));

    auto lbIndex = builder.create<mlir::arith::IndexCastOp>(
        loop.getLoc(), rewriter.getIndexType(), lb);

    auto ubIndex = builder.create<mlir::arith::IndexCastOp>(
        loop.getLoc(), rewriter.getIndexType(), ubLoad);

    auto step = builder.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 1);

    auto forOp =
        builder.create<mlir::scf::ForOp>(loop.getLoc(), lbIndex, ubIndex, step);

    rewriter.inlineRegionBefore(loop.getAfter(), forOp.getBody());

    auto yieldOp = forOp.getBody()->getTerminator();

    for (int i = 0; i < yieldOp->getNumOperands(); i++) {
      yieldOp->eraseOperand(i);
    }

    forOp.getBodyRegion().getBlocks().back().erase();

    // forOp.dump();

    loop.replaceAllUsesWith(forOp.getResults());

    loop.erase();

    return mlir::WalkResult::advance();
  });

  llvm::outs() << *module;
  return 0;

  funcOp.walk([&](mlir::scf::WhileOp loop) {
    if (loop.getOperation()->getParentOfType<mlir::scf::WhileOp>() != nullptr) {
      return mlir::WalkResult::skip();
    }

    llvm::json::Object loopObject;

    uint8_t loopNestSize = countLoopNestDepth(loop);

    auto innerLoops = getLoopsOnLevel(loop, loopNestSize - 1);

    uint8_t loopCounter = 0;

    llvm::errs() << "Collected inner loops \n";
    // this for is needed to handle edge cases where there are not perfectly
    // nested loops
    for (auto innerLoop : innerLoops) {
      auto arguments = getFuncArgumentsUsedInLoop(innerLoop);

      llvm::json::Array args;
      // for every function arguments used in the innermost loop
      for (auto arg : llvm::enumerate(arguments)) {
        args.push_back(getArgumentMetadata(arg.value().getValue(),
                                           toInt(arg.value().getKey())));
      }

      llvm::json::Object indexRange;
      indexRange["begin"] = llvm::json::Value(loopNestCounter);
      indexRange["end"] =
          llvm::json::Value(loopNestCounter + loopNestSize - 1 + loopCounter);
      loopObject["loop_position_index_range"] =
          llvm::json::Value(std::move(indexRange));
      loopObject["size"] = llvm::json::Value(loopNestSize);
      loopObject["args"] = llvm::json::Value(std::move(args));

      llvm::json::Array bounds;
      collectLoopBounds(innerLoop, bounds);
      loopObject["bounds"] = llvm::json::Value(std::move(bounds));

      loops.push_back(llvm::json::Value(std::move(loopObject)));
      loopCounter++;
    }

    loopNestCounter++;

    return mlir::WalkResult::advance();
  });

  function["loops"] = llvm::json::Value(std::move(loops));
  llvm::outs() << llvm::json::Value(std::move(function)) << "\n";

  return 0;
}
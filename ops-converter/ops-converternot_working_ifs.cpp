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

bool isWhileLoop(mlir::Operation *op) { return isa<mlir::scf::WhileOp>(op); }
bool isForLoop(mlir::Operation *op) { return isa<mlir::scf::ForOp>(op); }
bool isLoop(mlir::Operation *op) { return isWhileLoop(op) || isForLoop(op); }

mlir::Region *getLoopBody(mlir::Operation *op) {
  if (isWhileLoop(op)) {
    return &op->getRegion(1);
  } else if (isForLoop(op)) {
    return &op->getRegion(0);
  } else {
    llvm_unreachable("Not a loop");
  }
}

/**
 * @brief Returns all ops from the main loop region
 */
llvm::SmallVector<mlir::Operation *> getInnerLoopOps(mlir::Operation *op) {
  if (isWhileLoop(op)) {
    return llvm::SmallVector<mlir::Operation *>(
        cast<mlir::scf::WhileOp>(op).getAfter().getOps());
  } else if (isForLoop(op)) {
    return llvm::SmallVector<mlir::Operation *>(
        cast<mlir::scf::ForOp>(op).getLoopBody().getOps());
  } else {
    llvm_unreachable("Not a loop");
  }
}

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
  auto parentOp = arg.getParentRegion()->getParentOp();
  auto parentDepth =
      parentOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt();

  auto parentWhileOp =
      cast<mlir::scf::WhileOp>(arg.getParentRegion()->getParentOp());

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

        setLoopAttribute(bound, parentDepth);
        setInductionAttribute(bound);

        return bound;
      }

      // check if the block argument is a global memref value
      // in some case a global memref values is route through the scf before
      // region
      if (auto loadOp =
              dyn_cast<mlir::memref::LoadOp>(a.value().getDefiningOp())) {

        auto memref = loadOp.getMemRef();
        setLoopAttribute(memref, parentDepth);

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
uint8_t countLoopNestDepth(mlir::Operation *op, uint8_t depth = 1) {
  auto max = depth;

  if (isWhileLoop) {
    auto whileOp = cast<mlir::scf::WhileOp>(op);
    for (auto innerLoop :
         whileOp.getAfter().getBlocks().front().getOps<mlir::scf::WhileOp>()) {
      uint8_t nextDepth = countLoopNestDepth(innerLoop, depth + 1);

      if (nextDepth > max) {
        max = nextDepth;
      }
    }
  }

  if (isForLoop) {
    auto forOp = cast<mlir::scf::ForOp>(op);
    for (auto innerLoop :
         forOp.getRegion().getBlocks().front().getOps<mlir::scf::ForOp>()) {
      uint8_t nextDepth = countLoopNestDepth(innerLoop, depth + 1);

      if (nextDepth > max) {
        max = nextDepth;
      }
    }
  }

  return max;
}

// auto getLoopsOnLevel(mlir::Operation *op, uint8_t level, uint8_t depth = 0) {
//   mlir::SmallVector<mlir::Operation *> loops;

//   op->walk([&](mlir::Operation *childOp) {
//     if (isLoop(childOp)) {
//       if (level == op->getAttr("level").cast<mlir::IntegerAttr>().getInt()) {
//         loops.push_back(childOp);
//       }
//     }
//   });

//   return loops;
// }

llvm::StringMap<mlir::BlockArgument>
getFuncArgumentsUsedInLoop(mlir::Operation *loop) {
  llvm::StringMap<mlir::BlockArgument> argumentMap;

  loop->getRegions().back().walk([&](mlir::Operation *op) {
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

// mlir::WalkResult setLoopDepthAttribute(mlir::Operation *op, uint8_t depth) {

//   op->setAttr("depth", mlir::IntegerAttr::get(
//                            mlir::IntegerType::get(op->getContext(), 8),
//                            depth));

//   if (isWhileLoop(op)) {
//     auto loop = cast<mlir::scf::WhileOp>(op);
//     auto initialValue = loop.getInits()[0];

//     if (auto constantIntOp = dyn_cast<mlir::arith::ConstantIntOp>(
//             initialValue.getDefiningOp())) {
//       loop->setAttr("lb", constantIntOp.getValueAttr());
//     }

//     auto ubValue = cast<mlir::arith::CmpIOp>(
//                        loop.getConditionOp().getOperand(0).getDefiningOp())
//                        ->getOperand(1)
//                        .getDefiningOp()
//                        ->getOperand(0);

//     if (auto ubGlobal =
//             dyn_cast<mlir::memref::GetGlobalOp>(ubValue.getDefiningOp())) {

//       loop->setAttr("ub", ubGlobal.getNameAttr());
//     }
//   }

//   if (isForLoop(op)) {
//     auto loop = cast<mlir::scf::ForOp>(op);
//     auto initialValue = loop.getLowerBound();

//     if (auto constantIntOp = dyn_cast<mlir::arith::ConstantIndexOp>(
//             loop.getLowerBound().getDefiningOp())) {
//       loop->setAttr("lb", constantIntOp.getValueAttr());
//     }

//     if (auto constantIntOp = dyn_cast<mlir::arith::ConstantIndexOp>(
//             loop.getUpperBound().getDefiningOp())) {
//       loop->setAttr("ub", constantIntOp.getValueAttr());
//     }

//     loop.getBodyRegion().walk([&](mlir::Operation *inner) {
//       if (!isLoop(inner)) {
//         return mlir::WalkResult::skip();
//       }

//       return setLoopDepthAttribute(inner, depth + 1);
//     });
//   }

//   return mlir::WalkResult::advance();
// }

void collectLoopBounds(mlir::Operation *loop, llvm::json::Array &bounds) {

  auto lb = llvm::json::Value(
      Twine(loop->getAttr("lb").cast<mlir::IntegerAttr>().getInt()).str());
  auto ub = llvm::json::Value(
      loop->getAttr("ub").cast<mlir::FlatSymbolRefAttr>().getValue());
  bounds.insert(bounds.begin(), ub);
  bounds.insert(bounds.begin(), lb);

  if (auto parentWhileOp = loop->getParentOfType<mlir::scf::WhileOp>()) {
    collectLoopBounds(parentWhileOp, bounds);
  } else if (auto parentForOp = loop->getParentOfType<mlir::scf::ForOp>()) {
    collectLoopBounds(parentForOp, bounds);
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

  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    return setLoopDepthAttribute(op, 1);
  });

  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    // if op is a loop but it is not an outermost loop of a loop nest, skip
    if (op->getParentOfType<mlir::scf::WhileOp>() != nullptr ||
        op->getParentOfType<mlir::scf::ForOp>() != nullptr) {
      return mlir::WalkResult::skip();
    }

    llvm::json::Object loopObject;

    // uint8_t loopNestSize = countLoopNestDepth(op);

    // auto innerLoops = getLoopsOnLevel(op, loopNestSize - 1);

    uint8_t loopCounter = 0;

    llvm::errs() << "Collected inner loops \n";

    // this for is needed to handle edge cases where there are not perfectly
    // nested loops
    // for (auto innerLoop : innerLoops) {
    //   auto arguments = getFuncArgumentsUsedInLoop(innerLoop);

    //   llvm::json::Array args;
    //   // for every function arguments used in the innermost loop
    //   for (auto arg : llvm::enumerate(arguments)) {
    //     args.push_back(getArgumentMetadata(arg.value().getValue(),
    //                                        toInt(arg.value().getKey())));
    //   }

    //   llvm::json::Object indexRange;
    //   indexRange["begin"] = llvm::json::Value(loopNestCounter);
    //   indexRange["end"] =
    //       llvm::json::Value(loopNestCounter + loopNestSize - 1 +
    //       loopCounter);
    //   loopObject["loop_position_index_range"] =
    //       llvm::json::Value(std::move(indexRange));
    //   loopObject["size"] = llvm::json::Value(loopNestSize);
    //   // loopObject["args"] = llvm::json::Value(std::move(args));

    //   llvm::json::Array bounds;
    //   collectLoopBounds(innerLoop, bounds);
    //   loopObject["bounds"] = llvm::json::Value(std::move(bounds));

    //   loops.push_back(llvm::json::Value(std::move(loopObject)));
    //   loopCounter++;
    // }

    loopNestCounter++;

    return mlir::WalkResult::advance();
  });

  function["loops"] = llvm::json::Value(std::move(loops));
  llvm::outs() << llvm::json::Value(std::move(function)) << "\n";

  return 0;
}
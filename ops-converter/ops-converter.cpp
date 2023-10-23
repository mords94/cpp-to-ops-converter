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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Lib/ExpressionTree.h"
#include "Lib/helpers.h"

#include <algorithm>
#include <regex>
#include <string>

using namespace llvm;

std::string getLine(mlir::FileLineColLoc loc);
std::string getStrippedFileNameFromFileLineColLoc(mlir::FileLineColLoc loc);

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<kernel mlir file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<std::string> originalSource(cl::Positional,
                                           cl::desc("<original source file>"),
                                           cl::value_desc("path"),
                                           cl::Required);

static cl::opt<std::string> functionName("function", cl::desc("Function name"),
                                         cl::value_desc("string"),
                                         cl::Required);

static cl::opt<bool> enableDebug("opconv-debug", cl::desc("Enable debug"),
                                 cl::init(false));

static cl::opt<bool> enableDebugModule("opconv-debug-module",
                                       cl::desc("Print module after labelings"),
                                       cl::init(false));

static cl::opt<bool> enableAnnotateArgs("opconv-annotate-args",
                                        cl::desc("Print loop after labelings"),
                                        cl::init(false));

static cl::opt<bool>
    enableDebugConstants("opconv-debug-constants",
                         cl::desc("Stop if locally assigned constant is found"),
                         cl::init(false));

static cl::opt<bool>
    enableDetailedPrint("opconv-detailed",
                        cl::desc("Add loop and call orders for dag"),
                        cl::init(false));

static cl::opt<bool> emitMLIR("opconv-emit-mlir", cl::desc("Emit MLIR"),
                              cl::init(false));

bool isWhileLoop(mlir::Operation *op) { return isa<mlir::scf::WhileOp>(op); }
bool isForLoop(mlir::Operation *op) { return isa<mlir::scf::ForOp>(op); }
bool isLoop(mlir::Operation *op) { return isWhileLoop(op) || isForLoop(op); }

/**
 * Prints the line of the source and highlights the location with a ^ symbol
 * pretty print it with frames
 */

class Color {
public:
  enum class Value {
    red,
    green,
    blue,
    yellow,
  };

  Color(Value value) : value(value) {}

  std::string toString() const {
    switch (value) {
    case Value::red:
      return "\033[1;31m";
    case Value::green:
      return "\033[1;32m";
    case Value::blue:
      return "\033[1;34m";
    case Value::yellow:
      return "\033[1;33m";
    default:
      return "";
    }
  }

  Value value;
};

void printLoc(mlir::Location loc, Color color = Color::Value::red) {
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    // print frame and empty lines
    llvm::errs() << "\n" << std::string(80, '-') << "\n";

    auto line = fileLineColLoc.getLine();
    auto col = fileLineColLoc.getColumn();

    auto lineStr = getLine(fileLineColLoc);

    llvm::errs() << color.toString() << originalSource << ":" << line << ":"
                 << col << "\033[0m\n";
    llvm::errs() << lineStr << "\n";

    llvm::errs() << std::string(col - 1, ' ') << "\033[1;31m^\033[0m\n";

    llvm::errs() << std::string(80, '-') << "\n";
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

llvm::json::Value toJson(mlir::FileLineColLoc loc) {
  llvm::json::Object location;
  std::string fileName = getStrippedFileNameFromFileLineColLoc(loc);

  assert(fileName != "" && "File name is empty");

  location["file"] = llvm::json::Value(fileName);
  location["line"] = llvm::json::Value(loc.getLine());
  location["col"] = llvm::json::Value(loc.getColumn());
  location["source"] = llvm::json::Value(getLine(loc));

  return llvm::json::Value(std::move(location));
}

std::string getInductionVariableFromStatement(std::string forStmt) {

  std::vector<std::string> forStmtParts;
  std::string forStmtPart;
  std::istringstream forStmtStream(forStmt);
  while (std::getline(forStmtStream, forStmtPart, ';')) {
    forStmtParts.push_back(forStmtPart);
  }

  auto condition = forStmtParts[1];

  std::string inductionVariable = "";
  if (condition.find(">") != std::string::npos) {
    inductionVariable = condition.substr(0, condition.find(">"));
  } else if (condition.find("<") != std::string::npos) {
    inductionVariable = condition.substr(0, condition.find("<"));
  }

  inductionVariable.erase(std::remove_if(inductionVariable.begin(),
                                         inductionVariable.end(), isspace),
                          inductionVariable.end());

  return inductionVariable;
}

mlir::Operation *getParentLoop(mlir::Operation *op) {
  auto parent = op->getParentOp();
  while (parent != nullptr && isLoop(parent)) {
    return getParentLoop(parent);
  }

  return parent;
}

std::string getLine(mlir::FileLineColLoc loc) {
  auto line = getSource(loc.getFilename());

  auto lineNum = loc.getLine() - 1;

  if (lineNum >= line->size()) {
    return "";
  }

  return line->at(lineNum);
}

llvm::json::Object getClosingBracket(mlir::Location _loc) {
  llvm::json::Object location;
  if (auto loc = _loc.dyn_cast<mlir::FileLineColLoc>()) {
    std::string fileName = getStrippedFileNameFromFileLineColLoc(loc);

    assert(fileName != "" && "File name is empty");

    location["file"] = llvm::json::Value(fileName);

    int openBrackets = 1;
    int lineNum = loc.getLine() - 1;
    int colNum = loc.getColumn() - 1;

    auto line = getSource(loc.getFilename());

    // iterate through every line until the end of the file
    for (int i = lineNum; i < line->size(); i++) {
      auto currentLine = line->at(i);

      // iterate through every character until the end of the line
      for (int j = colNum; j < currentLine.size(); j++) {
        if (currentLine[j] == '{') {
          openBrackets++;
        } else if (currentLine[j] == '}') {
          openBrackets--;
        }

        if (openBrackets == 0) {
          location["line"] = llvm::json::Value(i + 1);
          location["col"] = llvm::json::Value(j + 1);
          location["source"] = llvm::json::Value(currentLine);
          return location;
        }
      }
      colNum = 0;
    }

    return location;
  }
}

std::string getInductionVariableFromLoop(mlir::Operation *op) {
  std::string stmt = getLine(op->getLoc().dyn_cast<mlir::FileLineColLoc>());

  return getInductionVariableFromStatement(stmt);
}

std::string getStrippedFileNameFromFileLineColLoc(mlir::FileLineColLoc loc) {
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    auto fileName = fileLineColLoc.getFilename();
    auto strippedFileName =
        fileName.str().substr(fileName.str().find_last_of("/") + 1);

    return strippedFileName;
  }
  return "";
}

llvm::json::Object getLoopAttrObject(mlir::Location loc) {
  llvm::json::Object attr;
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    llvm::json::Object location;
    std::string fileName =
        getStrippedFileNameFromFileLineColLoc(fileLineColLoc);

    assert(fileName != "" && "File name is empty");

    location["file"] = llvm::json::Value(fileName);
    location["line"] = llvm::json::Value(fileLineColLoc.getLine());
    location["col"] = llvm::json::Value(fileLineColLoc.getColumn());
    location["source"] = llvm::json::Value(getLine(fileLineColLoc));
    attr["induction_var"] = llvm::json::Value(
        getInductionVariableFromStatement(getLine(fileLineColLoc)));

    attr["location"] = llvm::json::Value(std::move(location));

    attr["close_location"] = llvm::json::Value(getClosingBracket(loc));
  }

  return attr;
}

llvm::json::Object getArgLocation(mlir::Location loc) {
  llvm::json::Object attr;
  if (auto fileLineColLoc = loc.dyn_cast<mlir::FileLineColLoc>()) {
    llvm::json::Object location;
    std::string fileName =
        getStrippedFileNameFromFileLineColLoc(fileLineColLoc);

    location["file"] = llvm::json::Value(fileName);
    location["line"] = llvm::json::Value(fileLineColLoc.getLine());
    location["col"] = llvm::json::Value(fileLineColLoc.getColumn());
    location["associated_line"] = llvm::json::Value(getLine(fileLineColLoc));

    attr["location"] = llvm::json::Value(std::move(location));
  }

  return attr;
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

  llvm::errs() << "   [buildTree] Building tree for value " << value << "\n";

  if (auto op = value.getDefiningOp()) {
    llvm::errs() << "   [buildTree] Bulding a new node with value on level "
                 << depth << "\n";

    if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(op)) {
      llvm::errs() << "   [buildTree] Load operation \n";

      // global memref
      node.setValue(loadOp.getMemRef());
    } else if (auto constantIntOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      llvm::errs() << "   [buildTree] Constant offset operation \n";

      // index offset
      node.setValue(constantIntOp.getResult());
    } else if (auto indexCastOp = dyn_cast<mlir::arith::IndexCastOp>(op)) {
      llvm::errs() << "   [buildTree] Index cast operation \n";

      // Node *children[1] = {new Node()};

      // node.setLeft(children[0]);
      // node.setValue(indexCastOp.getIn());

      buildTree(node, indexCastOp.getIn(), depth);

    } else if (op->getOperands().size() == 2) {
      llvm::errs() << "   [buildTree] Binary operation \n";
      // binary (addi or muli) operation
      Node *children[2] = {new Node(), new Node()};

      node.setLeft(children[0]);
      node.setRight(children[1]);
      node.setValue(value);
      for (auto en : llvm::enumerate(value.getDefiningOp()->getOperands())) {
        buildTree(*children[en.index()], en.value(), depth + 1);
      }
    } else {
      op->emitOpError("   [buildTree] Unknown operation while building memref "
                      "indexing expression tree");
    }
  } else if (auto blockArgument = dyn_cast<mlir::BlockArgument>(value)) {
    llvm::errs()
        << "   [buildTree] Bulding a new node with a block argument on level "
        << depth << "\n";

    // FIXME: finish rewrite pattern for this
    auto parent = blockArgument.getOwner()->getParentOp();

    if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(parent)) {
      auto conditionOperand =
          whileOp.getConditionOp().getArgs()[blockArgument.getArgNumber()];

      if (conditionOperand.isa<mlir::BlockArgument>()) {
        node.setOwnerLoop(blockArgument.getOwner()->getParentOp());
        node.setValue(value);
      } else {
        if (auto loadOp = dyn_cast<mlir::memref::LoadOp>(
                conditionOperand.getDefiningOp())) {
          node.setValue(loadOp.getMemRef());
        }
      }
    }
  }
};

/**
 * @brief Counts the maximum depth of nested loops within the given loop
 * operation
 * @param loopOp The loop operation to check
 * @param depth The current depth
 */
auto countLoopNestDepth(mlir::Operation *op, int64_t depth = 1) {
  auto max = depth;

  op->walk([&max](mlir::Operation *childOp) {
    if (!isLoop(childOp)) {
      return mlir::WalkResult::skip();
    }

    auto nextDepth =
        childOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt();

    if (nextDepth > max) {
      max = nextDepth;
    }

    return mlir::WalkResult::advance();
  });

  return max;
}

auto getLoopsOnLevel(mlir::Operation *op, int64_t level) {
  mlir::SmallVector<mlir::Operation *> loops;
  op->walk([&](mlir::Operation *childOp) {
    if (!isLoop(childOp)) {
      return;
    }

    if (level == childOp->getAttr("depth").cast<mlir::IntegerAttr>().getInt()) {
      loops.push_back(childOp);
    }
  });

  return loops;
}

llvm::StringMap<mlir::BlockArgument>
getFuncArgumentsUsedInLoop(mlir::Operation *loop) {
  llvm::StringMap<mlir::BlockArgument> argumentMap;

  loop->getRegions().back().walk([&](mlir::Operation *op) {
    if (isa<mlir::memref::LoadOp>(op) || isa<mlir::memref::StoreOp>(op)) {
      llvm::errs()
          << "   [getFuncArgumentsUsedInLoop] Found load or store op \n";
      mlir::Value memref =
          op->getOperand(isa<mlir::memref::LoadOp>(op) ? 0 : 1);

      if (memref.isa<mlir::BlockArgument>()) {
        llvm::errs() << "   [getFuncArgumentsUsedInLoop] " << op->getName()
                     << " block argument \n";
        mlir::BlockArgument arg = memref.cast<mlir::BlockArgument>();
        argumentMap[Twine(arg.getArgNumber()).str()] = arg;
      } else {
        llvm::errs() << "   [getFuncArgumentsUsedInLoop] " << op->getName()
                     << " memref \n";
        memref.dump();
      }
    } else {

      const bool hasFunctionArgumentAsOperand =
          llvm::any_of(op->getOperands(), [&](mlir::Value value) {
            return isFunctionArgument(value);
          });

      if (hasFunctionArgumentAsOperand) {
        op->emitError("   [getFuncArgumentsUsedInLoop] Argument is used by an "
                      "unsupported operation " +
                      op->getName().getStringRef());
      }
    }
  });

  return argumentMap;
}

mlir::Operation *getParentLoopOperation(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();

  while (parent != nullptr) {
    if (isLoop(parent)) {
      return parent;
    }

    parent = parent->getParentOp();
  }

  return nullptr;
}

template <typename T>
llvm::SmallVector<T> filterStoreUses(llvm::SmallVector<T> &argStoreUses,
                                     int loopIndex) {
  llvm::SmallVector<T> filteredStoreUses;
  llvm::errs()
      << "       [filterStoreUses] Filtering store uses for loop index "
      << loopIndex << "\n";
  for (auto use : argStoreUses) {
    auto parentLoop = getParentLoopOperation(use);

    if (parentLoop == nullptr) {
      continue;
    }

    if (parentLoop->template getAttrOfType<mlir::IntegerAttr>("loop_index")
            .getInt() == loopIndex) {
      filteredStoreUses.push_back(use);
    }
  }

  return filteredStoreUses;
}

template <typename T>
void setReadOrStoreValues(mlir::BlockArgument arg,
                          llvm::json::Object &jsonContainer, int loopIndex,
                          llvm::StringRef key) {

  if (has_use<T>(arg)) {
    llvm::errs() << "     [setReadOrStoreValues] Found use of argument (" << key
                 << ")\n";
    llvm::SmallVector<T> argStoreUses = get_use<T>(arg);

    llvm::errs() << "     [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ")\n";

    argStoreUses = filterStoreUses<T>(argStoreUses, loopIndex);

    llvm::errs() << "     [setReadOrStoreValues] Found " << argStoreUses.size()
                 << " uses of argument (" << key << ") on loop level "
                 << loopIndex << "\n";

    // we use flat 1d index for 2D and 3D arrays therefore it will be  only
    // one index
    llvm::json::Array arrOfUseIndices;
    jsonContainer[key.str() + std::string("_count")] =
        llvm::json::Value((int64_t)argStoreUses.size());

    for (auto use : llvm::enumerate(argStoreUses)) {
      llvm::json::Object useDescription;

      auto index = use.value().getIndices()[0];

      llvm::errs() << "     [setReadOrStoreValues] Building expression tree\n";
      Node *tree = new Node();

      auto isIndexOp = index.getDefiningOp() != nullptr &&
                       isa<mlir::arith::IndexCastOp>(index.getDefiningOp());

      buildTree(*tree, isIndexOp ? index.getDefiningOp()->getOperand(0) : index,
                0);

      llvm::errs()
          << "      [setReadOrStoreValues] End of Building expression tree\n";

      if (enableDebug) {
        tree->dump();
        argStoreUses[0].dump();
      }

      if (enableDebug) {
        useDescription["debug"] = tree->toDot();
      }

      useDescription["indexes"] = llvm::json::Value(tree->getAllIndexes());
      useDescription["dims"] = llvm::json::Value(tree->getMemrefSize());

      arrOfUseIndices.push_back(llvm::json::Value(std::move(useDescription)));
    }
    jsonContainer[(key.str() + std::string("_offsets"))] =
        llvm::json::Value(std::move(arrOfUseIndices));
    jsonContainer[key] = llvm::json::Value(argStoreUses.size() > 0);
  } else {
    llvm::errs() << "     [setReadOrStoreValues] No use of argument " << key
                 << "\n";
    jsonContainer[key] = llvm::json::Value(false);
  }
}

std::string getLoopArgName(mlir::BlockArgument arg) {
  std::string result = "";
  llvm::raw_string_ostream os(result);

  auto fileLineCol = arg.getLoc().cast<mlir::FileLineColLoc>();

  std::string line = getLine(fileLineCol);

  int initialColumn = fileLineCol.getColumn() - 1;
  // bool hit = false;

  for (int i = initialColumn; i < line.size(); i++) {
    if (line[i] == ';' || line[i] == ',') {
      break;
    }
    os << line[i];
  }

  return os.str();
}

/*
2nd FIXME: Is this comment written by me? If so, I should rewrite it.
TODO: This is a hacky way to get the name of the argument. We should
      find a better way to do this.
      For example, we can use the name of the function argument but we need to
      merge the rows of the function argument declarations.
*/
std::string getArgNameFromFuncLocation(mlir::BlockArgument arg) {

  std::string result = "";
  llvm::raw_string_ostream os(result);

  auto firstUseOfArg = arg.getUses().begin().getUser();
  auto fileLineCol = firstUseOfArg->getLoc().cast<mlir::FileLineColLoc>();
  std::string line = getLine(fileLineCol);

  auto plusEqual = line[fileLineCol.getColumn() - 1] == '+' &&
                   line[fileLineCol.getColumn()] == '=';

  auto minusEqual = line[fileLineCol.getColumn() - 1] == '-' &&
                    line[fileLineCol.getColumn()] == '=';

  auto equal = line[fileLineCol.getColumn() - 1] == '=';

  auto initialColumn =
      plusEqual || minusEqual || equal ? 0 : fileLineCol.getColumn() - 1;

  bool hit = false;
  for (int i = initialColumn; i < line.size(); i++) {
    if (hit && line[i] == '*') {
      break;
    }

    if (line[i] == ' ' || line[i] == '*' || line[i] == '(') {
      continue;
    }
    if (line[i] == '[' || line[i] == '=' || line[i] == ';' || line[i] == ',' ||
        line[i] == ')' || line[i] == '!' || line[i] == '<' || line[i] == '>') {
      break;
    }
    os << line[i];
    hit = true;
  }

  result.erase(std::remove_if(result.begin(), result.end(), isspace),
               result.end());

  return result;
}

llvm::json::Object getArgumentMetadata(mlir::BlockArgument arg, int argIndex,
                                       int loopIndex) {

  llvm::json::Object argObject;

  std::string backupName = "arg" + Twine(arg.getArgNumber()).str();

  std::string argName = getArgNameFromFuncLocation(arg);

  if (argName == "") {
    argName = backupName;
  }

  argObject["arg_index"] = llvm::json::Value(argIndex);

  argObject["name"] = llvm::json::Value(argName);
  argObject["meta"] = llvm::json::Value(
      getArgLocation(arg.getUses().begin().getUser()->getLoc()));

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"]
               << " write\n";
  setReadOrStoreValues<mlir::memref::StoreOp>(arg, argObject, loopIndex,
                                              "write");

  llvm::errs() << "   [getArgumentMetadata] " << argObject["name"] << " read\n";
  setReadOrStoreValues<mlir::memref::LoadOp>(arg, argObject, loopIndex, "read");

  return argObject;
}

mlir::StringAttr convertAttributeToStringAttr(mlir::Attribute attr,
                                              mlir::Builder &builder) {
  return llvm::TypeSwitch<mlir::Attribute, mlir::StringAttr>(attr)
      .Case<mlir::StringAttr>([&](mlir::StringAttr stringAttr) {
        // Attribute is already a StringAttr, no conversion needed
        return stringAttr;
      })
      .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr intAttr) {
        // Convert IntegerAttr to StringAttr
        mlir::APInt value = intAttr.getValue();

        llvm::SmallString<32> str;
        value.toString(str, /*radix=*/10, /*signed=*/true);
        return builder.getStringAttr(str);
      })
      .Case<mlir::BoolAttr>([&](mlir::BoolAttr boolAttr) {
        // Convert BoolAttr to StringAttr
        return builder.getStringAttr(boolAttr.getValue() ? "true" : "false");
      })
      .Case<mlir::FloatAttr>([&](mlir::FloatAttr floatAttr) {
        // Convert FloatAttr to StringAttr
        llvm::SmallString<32> str;
        floatAttr.getValue().toString(str);
        return builder.getStringAttr(str);
      })
      .Case<mlir::FlatSymbolRefAttr>([&](mlir::FlatSymbolRefAttr symbolAttr) {
        return builder.getStringAttr(symbolAttr.getValue().str());
      })
      .Default([&](mlir::Attribute) {
        llvm::errs() << "Unsupported attribute type: " << attr << "\n";
        return builder.getStringAttr("Unsupported");
      });
}

mlir::StringAttr replaceStringInAttr(mlir::StringAttr attr,
                                     mlir::StringRef oldValue,
                                     mlir::StringRef newValue,
                                     mlir::MLIRContext *context) {
  std::string originalStr = attr.getValue().str();
  size_t pos = originalStr.find(oldValue.str());
  if (pos != std::string::npos) {
    originalStr.replace(pos, oldValue.size(), newValue.str());
    return mlir::StringAttr::get(context, originalStr);
  }
  return attr;
}

mlir::StringAttr mergeAttributes(mlir::Attribute attr1, mlir::Attribute attr2,
                                 llvm::StringRef glue,
                                 mlir::MLIRContext *context) {

  auto builder = mlir::Builder(context);
  auto attr1Str = convertAttributeToStringAttr(attr1, builder);
  auto attr2Str = convertAttributeToStringAttr(attr2, builder);

  auto resultStr = attr1Str.str() + glue + attr2Str.str();

  return builder.getStringAttr(resultStr);
}

mlir::Attribute getBoundAttribute(mlir::Value boundValue) {
  mlir::Attribute boundAttr;

  llvm::errs() << "Inside getBoundAttribute\n";

  boundValue.dump();

  // if block argument
  if (isa<mlir::BlockArgument>(boundValue)) {
    llvm::errs() << "Inside getBoundAttribute block argument\n";
    auto blockArg = cast<mlir::BlockArgument>(boundValue);
    blockArg.getOwner()->dump();
  }

  llvm::TypeSwitch<mlir::Operation *>(boundValue.getDefiningOp())
      .Case([&](mlir::arith::ConstantOp constantOp) {
        boundAttr = constantOp.getValueAttr();
      })
      .Case([&](mlir::arith::IndexCastOp indexCastOp) {
        auto lbValueOperand =
            indexCastOp.getIn().getDefiningOp()->getOperand(0);

        boundAttr = getBoundAttribute(lbValueOperand);
      })
      .Case([&](mlir::memref::LoadOp lbLoad) {
        boundAttr = getBoundAttribute(lbLoad.getMemRef());
      })
      .Case([&](mlir::memref::GetGlobalOp lbGlobal) {
        boundAttr = lbGlobal.getNameAttr();
      })
      // addIndexOp
      .Case([&](mlir::arith::AddIOp addOp) {
        auto lhsAttr = getBoundAttribute(addOp.getLhs());
        auto rhsAttr = getBoundAttribute(addOp.getRhs());

        auto merged =
            mergeAttributes(lhsAttr, rhsAttr, "+", addOp.getContext());

        // replace "+-" with "-" if the offset is -
        boundAttr = replaceStringInAttr(merged, "+-", "-", addOp.getContext());
      })
      .Default([&](mlir::Operation *op) {
        llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                     << "Loop has non-constant lower bound\n";

        auto valueAsString =
            boundValue.getDefiningOp()->getName().getStringRef();
        boundAttr = mlir::StringAttr::get(op->getContext(), valueAsString);

        return mlir::WalkResult::interrupt();
      });

  return boundAttr;
}

int getArgIndexUsedInCmpi(mlir::Operation *cmpiOp) {
  // Get the block containing the cmpi operation
  auto *parentBlock = cmpiOp->getBlock();

  // Get the operand used in the cmpi operation.
  // Assuming that the operand at index 0 is the one you're interested in.
  mlir::Value operandUsed = cmpiOp->getOperand(0);

  // Iterate over the arguments in the parent block.
  for (int i = 0; i < parentBlock->getNumArguments(); ++i) {
    if (parentBlock->getArgument(i) == operandUsed) {
      return i;
    }
  }

  // If the operand was not found among the block arguments, return -1.
  return -1;
}

// TODO: Consider rewrite this completely:
// Since we have the debug information and it is already parsed to obtain the
// induction variable we can use it to get the loop bounds also.
mlir::WalkResult setLoopDepthAndBoundsAttributes(mlir::Operation *op,
                                                 uint64_t depth) {

  // assert is loop
  if (!isLoop(op)) {
    llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                 << "ERRO! Operation is not a loop\n";
    return mlir::WalkResult::interrupt();
  }

  // if has depth attribute, it means that it was already visited

  auto inductionVariable = getInductionVariableFromLoop(op);

  op->setAttr("depth", mlir::IntegerAttr::get(
                           mlir::IntegerType::get(op->getContext(), 8), depth));

  op->setAttr("induction_variable",
              mlir::StringAttr::get(op->getContext(), inductionVariable));

  mlir::Value lbValue;
  mlir::Value ubValue;
  mlir::Region *loopBodyRegion;

  bool increment;

  auto loop = op;

  if (isWhileLoop(op)) {
    auto whileLoop = cast<mlir::scf::WhileOp>(op);
    lbValue = whileLoop.getInits().back();

    auto cmpiOp = cast<mlir::arith::CmpIOp>(
        whileLoop.getConditionOp().getOperand(0).getDefiningOp());

    auto inductionArgIndex = getArgIndexUsedInCmpi(cmpiOp);

    assert(inductionArgIndex != -1 &&
           "      [setLoopDepthAndBoundsAttributes] "
           "ERRO! Induction variable not found in cmpi operation");

    lbValue = whileLoop.getInits()[inductionArgIndex];

    loopBodyRegion = &whileLoop.getAfter();

    ubValue = cast<mlir::arith::CmpIOp>(
                  whileLoop.getConditionOp().getOperand(0).getDefiningOp())
                  ->getOperand(1);

    llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
                 << "condition: " << whileLoop.getConditionOp().getOperand(0)
                 << "\n";

    // if condition compare is sgt
    if (auto cmp = dyn_cast<mlir::arith::CmpIOp>(
            whileLoop.getConditionOp().getOperand(0).getDefiningOp())) {
      if (cmp.getPredicate() == mlir::arith::CmpIPredicate::sgt) {
        increment = false;
      } else {
        increment = true;
      }
    }
  }

  if (isForLoop(op)) {
    auto forLoop = cast<mlir::scf::ForOp>(op);
    lbValue = forLoop.getLowerBound();
    ubValue = forLoop.getUpperBound();
    loopBodyRegion = &forLoop.getLoopBody();
    increment = true;
  }

  llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
               << "setting step attribute for op: " << op->getName() << "\n";

  loop->setAttr("step", mlir::IntegerAttr::get(
                            mlir::IntegerType::get(op->getContext(), 8),
                            increment ? 1 : -1));

  llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
               << "setting lb attributes for op: " << op->getName() << "\n";

  loop->setAttr("lb", getBoundAttribute(lbValue));

  llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
               << "setting ub attributes for op: " << op->getName() << "\n";
  loop->setAttr("ub", getBoundAttribute(ubValue));

  llvm::errs() << "      [setLoopDepthAndBoundsAttributes] "
               << "setting depth attribute for op: " << op->getName() << "\n";

  loopBodyRegion->walk([&](mlir::Operation *inner) {
    if (!isLoop(inner)) {
      return mlir::WalkResult::skip();
    }

    return setLoopDepthAndBoundsAttributes(inner, depth + 1);
  });

  return mlir::WalkResult::advance();
}

llvm::json::Value attributeToJson(mlir::Attribute attribute) {
  llvm::json::Value jsonValue = llvm::json::Value("<null>");

  assert(attribute && "Attribute is null");

  llvm::TypeSwitch<mlir::Attribute>(attribute)
      .Case([&](mlir::IntegerAttr attr) {
        jsonValue = llvm::json::Value(attr.getInt());
      })
      .Case([&](mlir::FlatSymbolRefAttr attr) {
        jsonValue = llvm::json::Value(attr.getValue());
      })
      .Case([&](mlir::StringAttr attr) {
        jsonValue = llvm::json::Value(attr.getValue());
      })
      .Default([&](mlir::Attribute attr) {
        llvm::errs() << "Unknown attribute type: " << attr << "\n";
      });

  return jsonValue;
}

std::string trim(const std::string &str) {
  size_t start = str.find_first_not_of(" \t\n\r\f\v");
  size_t end = str.find_last_not_of(" \t\n\r\f\v");

  if (start == std::string::npos) {
    return "";
  }

  return str.substr(start, end - start + 1);
}

std::string printOpAttributes(mlir::Operation *op) {
  std::string result = "";
  llvm::raw_string_ostream stream(result);
  op->getAttrDictionary().print(stream);

  return stream.str();
}

void collectLoopInfo(mlir::Operation *loop, llvm::json::Array &bounds,
                     llvm::json::Array &loopAttrs, llvm::json::Array &steps) {
  llvm::errs() << "    [collectLoopBounds] Attributes"
               << printOpAttributes(loop) << "\n";
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds lb\n";

  auto lb = loop->getAttr("lb");

  bounds.insert(bounds.end(), attributeToJson(lb));
  llvm::errs() << "   [collectLoopBounds] Collecting loop bounds ub\n";

  auto ub = loop->getAttr("ub");

  bounds.insert(bounds.end(), attributeToJson(ub));

  llvm::errs() << "   [collectLoopBounds] Collecting step\n";
  auto step = loop->getAttr("step");

  steps.insert(steps.end(), attributeToJson(step));

  llvm::errs()
      << "   [collectLoopBounds] Collecting loop bounds induction_variable\n";

  loopAttrs.insert(loopAttrs.end(),
                   llvm::json::Value(getLoopAttrObject(loop->getLoc())));

  if (auto parentWhileOp = loop->getParentOfType<mlir::scf::WhileOp>()) {
    collectLoopInfo(parentWhileOp, bounds, loopAttrs, steps);
  } else if (auto parentForOp = loop->getParentOfType<mlir::scf::ForOp>()) {
    collectLoopInfo(parentForOp, bounds, loopAttrs, steps);
  }
}

mlir::Value getInductionFromWhileGeneric(mlir::scf::WhileOp whileOp) {

  auto yieldOp = whileOp.getYieldOp();

  for (auto operand : yieldOp.getOperands()) {
    if (isa<mlir::arith::AddIOp>(operand.getDefiningOp())) {
      auto addIOp = dyn_cast<mlir::arith::AddIOp>(operand.getDefiningOp());

      auto lhs = addIOp.getOperand(0);
      auto rhs = addIOp.getOperand(1);

      if (lhs.isa<mlir::BlockArgument>() &&
          isa<mlir::arith::ConstantOp>(rhs.getDefiningOp()) &&
          dyn_cast<mlir::arith::ConstantIntOp>(rhs.getDefiningOp())
                  .getValue()
                  .cast<mlir::IntegerAttr>()
                  .getInt() == 1) {
        return lhs;
      }
    }
  }
}

mlir::Value getInductionFromWhile(mlir::scf::WhileOp whileOp) {

  auto yieldOp = whileOp.getYieldOp();

  auto firstOperand = yieldOp.getOperand(0).getDefiningOp();

  // assuming the first operand of yield is the result of i++
  auto firstOperandAsAddIOp =
      dyn_cast_or_null<mlir::arith::AddIOp>(firstOperand);

  assert(firstOperandAsAddIOp && "First operand of yield is not an addi op");

  // iterate operands of add op and return the one that is blockArg
  for (auto operand : firstOperandAsAddIOp.getOperands()) {
    if (operand.isa<mlir::BlockArgument>()) {
      return operand;
    }
  }

  return nullptr;
}

auto getLoopArgNameByTheFirstUseLoc(mlir::Value arg) {
  auto firstUse = arg.getUses().begin().getUser();

  auto loc = firstUse->getLoc().cast<mlir::FileLineColLoc>();

  auto line = getLine(loc);

  auto column = loc.getColumn() - 1;

  std::string result = "";

  llvm::raw_string_ostream os(result);

  for (int i = column; i < line.size(); i++) {
    if (line[i] == ';' || line[i] == ',') {
      break;
    }
    os << line[i];
  }

  return os.str();
}

std::optional<llvm::json::Object>
parseDeclaration(const std::string &declaration) {

  // std::regex
  // pattern(R"(^\s*(float|double)\s+\w+\s*=\s*\d+(\.\d+)?(f)?;\s*$)");
  std::regex varDeclPattern(
      R"(^\s*([\w:<>]+)\s+([_a-zA-Z]\w*)(\s*=\s*([^;]+))?;\s*$)");

  std::regex varAssignmentPattern(R"(^\s*(\w+)\s*=\s*(.*);$)");

  std::smatch match;
  llvm::json::Object declarationObject;
  std::string type, name, initialValue;

  if (std::regex_match(declaration, match, varDeclPattern)) {
    type = match[1];
    name = match[2];
    initialValue = match[4];

  } else if (std::regex_match(declaration, match, varAssignmentPattern)) {
    name = match[1];
    type = "unknown";
    initialValue = match[2];
  } else {
    llvm::errs() << "Declaration is not valid. Skip\n";
    llvm::errs() << "Declaration: " << declaration << "\n";

    return std::nullopt;
  }

  llvm::errs() << "[parseDeclaration] name_out: " << name << "\n";
  llvm::errs() << "[parseDeclaration] declaration: " << declaration << "\n";

  declarationObject["name"] = llvm::json::Value(name);
  declarationObject["initialValue"] = llvm::json::Value(initialValue);
  declarationObject["type_parsed"] = llvm::json::Value(type);

  return declarationObject;
}

llvm::json::Object mergeJsonObjects(llvm::json::Object &o1,
                                    llvm::json::Object o2) {
  llvm::json::Object merged;

  for (auto &pair : o1) {
    merged[pair.first] = pair.second;
  }

  for (auto &pair : o2) {
    merged[pair.first] = pair.second;
  }

  return merged;
}

bool isNumber(const std::string &str) {
  std::istringstream iss(str);
  double tempDouble;
  iss >> tempDouble;
  return !iss.fail() && iss.eof();
}

bool isOperator(char ch) {
  switch (ch) {
  case '+':
  case '-':
  case '*':
  case '/':
  case '%':
  case '^':
  case '&':
  case '|':
    return true;
  default:
    return false;
  }
}

bool isNumberSubstring(const std::string &str, size_t startPos) {
  size_t pos = startPos;
  bool seenDecimalPoint = false;
  bool seenSign = false; // To handle both '+' and '-'
  bool seenExponent = false;
  bool seenDigit = false;
  bool seenF = false;

  while (pos < str.size()) {
    char ch = str[pos];

    if (std::isspace(ch) || (isOperator(ch) && pos != startPos)) {
      break;
    }

    if (ch == 'f' && !seenF && seenDigit) {
      seenF = true;
      pos++; // Increment the position to skip 'f'
      break; // Exit the loop if 'f' is encountered
    } else if (ch == '.' && !seenDecimalPoint) {
      seenDecimalPoint = true;
    } else if ((ch == '-' || ch == '+') && !seenSign && !seenDigit &&
               (!seenExponent || str[pos - 1] == 'e' || str[pos - 1] == 'E')) {
      seenSign = true;
    } else if ((ch == 'e' || ch == 'E') && !seenExponent) {
      if (pos == startPos || !seenDigit) {
        return false;
      }
      seenExponent = true;
      seenSign =
          false; // Reset sign after 'e' or 'E' to allow for signs in exponent
    } else if (std::isdigit(ch)) {
      seenDigit = true;
    } else {
      return false;
    }
    pos++;
  }
  return seenDigit && pos > startPos; // Ensure that at least one digit was seen
}

void collectLocalConstants(mlir::Operation *op, llvm::json::Array &constants) {
  if (!isWhileLoop(op)) {
    return;
  }

  auto loopLineNumber = op->getLoc().cast<mlir::FileLineColLoc>().getLine();

  auto whileOp = cast<mlir::scf::WhileOp>(op);

  // for
  for (auto &op : whileOp.getAfter().front()) {

    bool isArithmetic =
        isa<mlir::arith::AddFOp>(op) || isa<mlir::arith::MulFOp>(op) ||
        isa<mlir::arith::SubFOp>(op) || isa<mlir::arith::DivFOp>(op);

    bool isMath = isa<mlir::math::SinOp>(op) || isa<mlir::math::CosOp>(op) ||
                  isa<mlir::math::ExpOp>(op) || isa<mlir::math::LogOp>(op) ||
                  isa<mlir::math::AtanOp>(op) || isa<mlir::math::TanhOp>(op) ||
                  isa<mlir::math::Atan2Op>(op) || isa<mlir::math::PowFOp>(op);

    if (isArithmetic || isMath) {
      llvm::errs() << "Computing operation: " << op << "\n";
      // iterate over the operands
      for (auto operand : op.getOperands()) {
        if (!operand.getDefiningOp()) {
          continue;
        }

        if (operand.getDefiningOp()->getParentOfType<mlir::func::FuncOp>() ||
            isa<mlir::arith::ConstantOp>(operand.getDefiningOp())) {
          auto operandOp = operand.getDefiningOp();
          // if constant is float
          // store the loc
          auto loc = operandOp->getLoc().cast<mlir::FileLineColLoc>();
          auto line = getLine(loc);

          if (loopLineNumber < loc.getLine()) {
            // filter out inline constants e.g., 2.0f .1 etc..
            if (isNumberSubstring(line, loc.getColumn() - 1)) {
              llvm::errs() << "Constant is inline.\n";
              printLoc(loc, Color::Value::yellow);
              continue;
            }

            if (line.substr(0, loc.getColumn() - 1).find('=') !=
                std::string::npos) {
              // to not fail for folded math expressions
              printLoc(loc, Color::Value::blue);

              continue;
            }

            llvm::errs()
                << "Constant or expression is defined inside the loop.\n";
            llvm::errs() << "Constant or expression: " << operandOp << "\n";

            printLoc(loc);
            if (enableDebugConstants) {
              exit(-1);
            }
          }

          llvm::json::Object constantObject;

          std::string type;
          llvm::raw_string_ostream rso(type);

          operandOp->getResult(0).getType().print(rso);

          auto decl = parseDeclaration(line);

          constantObject["type"] = llvm::json::Value(type);

          if (decl) {
            constantObject["location"] = llvm::json::Value(toJson(loc));
            llvm::json::Object mergedConstantObject =
                mergeJsonObjects(decl.value(), constantObject);

            constants.push_back(
                llvm::json::Value(std::move(mergedConstantObject)));
          } else {
            llvm::errs()
                << "DeclarationParseError: Failed to parse declaration\n";
            llvm::errs() << "DeclarationParseError: " << line << "\n";
            llvm::errs() << "DeclarationParseError: " << std::string(8, ' ')
                         << "^\n";
          }
        }
      }
    }
  }
}

/** This function called collectLocals that collects the local variables
 * used in the loop nest and stores them in a json object
 */
void collectLocals(mlir::Operation *op, llvm::json::Array &locals) {

  if (!isWhileLoop(op)) {
    return;
  }

  auto whileOp = cast<mlir::scf::WhileOp>(op);

  auto afterArgs = whileOp.getAfterArguments();

  auto inductionArg = getInductionFromWhileGeneric(whileOp);

  // get the terminator op (scf yield) from the after region

  for (auto arg : afterArgs) {
    // Skip induction vars
    if (arg == inductionArg) {
      continue;
    }

    llvm::json::Object localObject;

    llvm::errs() << "    [collectLocals] arg: " << arg << "\n";

    auto name = getLoopArgName(arg);

    auto loc = arg.getLoc().cast<mlir::FileLineColLoc>();

    auto declarationLine = getLine(loc);

    std::string str;
    llvm::raw_string_ostream rso(str);
    arg.getType().print(rso);

    localObject["index"] = llvm::json::Value(arg.getArgNumber());
    localObject["location"] = llvm::json::Value(toJson(loc));
    localObject["_type"] = llvm::json::Value(rso.str());

    auto declaration = parseDeclaration(declarationLine);

    if (!declaration) {
      continue;
    }

    auto merged = mergeJsonObjects(localObject, declaration.value());

    locals.push_back(llvm::json::Value(std::move(merged)));
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

  if (!funcOp) {
    llvm::errs() << "Function " << functionName.getValue() << " not found\n";

    module->dump();
    return -1;
  }

  /**
   * Pass for remove all functions except the one we are interested in
   */
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

  int loopIndex = 0;

  /**
   * Pass for annotating loops with index and LB and UB
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    op->setAttr("loop_index",
                mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 32),
                                       loopIndex++));

    return setLoopDepthAndBoundsAttributes(op, 1);
  });

  /**
   * Pass for swapping bounds of adds where the operands are an
   * induction variable and a constant to ensure that the right child is the
   * constant
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not an add operation, skip
    if (!isa<mlir::arith::AddIOp>(op)) {
      return mlir::WalkResult::skip();
    }

    auto addOp = cast<mlir::arith::AddIOp>(op);

    // if any of the operands is a block argument, skip
    if (addOp.getLhs().isa<mlir::BlockArgument>() ||
        addOp.getRhs().isa<mlir::BlockArgument>()) {
      return mlir::WalkResult::skip();
    }

    // if rhs is not a constant, skip
    if (!isa<mlir::arith::ConstantOp>(addOp.getRhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    // if lhs is not a muli operation, skip
    if (!isa<mlir::arith::MulIOp>(addOp.getLhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    auto builder = mlir::OpBuilder::atBlockBegin(addOp->getBlock());
    builder.setInsertionPoint(addOp);

    auto newOp = builder.create<mlir::arith::AddIOp>(
        addOp.getLoc(), addOp.getRhs(), addOp.getLhs());

    addOp.replaceAllUsesWith(newOp.getResult());

    addOp.erase();

    return mlir::WalkResult::advance();
  });

  /**
   * Pass for swapping bounds of multiplications where the operands are an
   * induction variable and a constant to ensure that the right child is the
   * constant
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not an muli operation, skip
    if (!isa<mlir::arith::MulIOp>(op)) {
      return mlir::WalkResult::skip();
    }

    auto mulOp = cast<mlir::arith::MulIOp>(op);

    // if rhs is not a constant, skip
    if (mulOp.getRhs().isa<mlir::BlockArgument>() ||
        !isa<mlir::arith::ConstantOp>(mulOp.getRhs().getDefiningOp())) {
      return mlir::WalkResult::skip();
    }

    auto builder = mlir::OpBuilder::atBlockBegin(mulOp->getBlock());
    builder.setInsertionPoint(mulOp);

    auto newOp = builder.create<mlir::arith::MulIOp>(
        mulOp.getLoc(), mulOp.getRhs(), mulOp.getLhs());

    mulOp.replaceAllUsesWith(newOp.getResult());

    mulOp.erase();

    return mlir::WalkResult::advance();
  });

  /**
   * Pass for annotating loop args with debug information
   */
  funcOp.walk([&](mlir::Operation *op) {
    if (!isWhileLoop(op)) {
      return mlir::WalkResult::skip();
    }

    auto whileOp = cast<mlir::scf::WhileOp>(op);

    auto afterArgs = whileOp.getAfterArguments();

    int index = 1;
    for (auto arg : afterArgs) {
      auto name = getLoopArgName(arg);

      auto loc = arg.getLoc().cast<mlir::FileLineColLoc>();

      auto declarationLine = getLine(loc);

      llvm::errs() << "[main] declarationLine: " << declarationLine << "\n";
      // save the line from col and set to a string attribute to the arg

      auto truncatedLine = declarationLine.substr(loc.getColumn() - 1);
      // name = arg#{index}

      auto attr = mlir::StringAttr::get(op->getContext(), truncatedLine);
      op->setAttr("arg" + Twine(index++).str(), attr);

      index++;
    }

    return mlir::WalkResult::advance();
  });

  if (emitMLIR) {
    module->print(llvm::outs());

    return 0;
  }

  int _index = 0;
  /**
   * Main pass for collecting metadata about loops
   */
  funcOp.walk([&](mlir::Operation *op) {
    // if operation is not a loop, skip
    if (!isLoop(op)) {
      return mlir::WalkResult::skip();
    }

    llvm::errs() << "[main] Checking whether it is an outermost loop... \n";

    // if op is a loop but it is not an outermost loop of a loop nest, skip
    if (op->getParentOfType<mlir::scf::WhileOp>() != nullptr ||
        op->getParentOfType<mlir::scf::ForOp>() != nullptr) {
      return mlir::WalkResult::skip();
    }

    llvm::errs() << "[main] Counting loop nest depth... \n";
    llvm::json::Object loopObject;

    auto loopNestSize = countLoopNestDepth(op);

    llvm::errs() << "[main] Collecting inner loops for nest with size: "
                 << Twine(loopNestSize).str() << "... \n";
    auto innerLoops = getLoopsOnLevel(op, loopNestSize);

    uint64_t loopCounter = 0;

    llvm::errs() << "[main] Collected inner loops: " << innerLoops.size()
                 << "\n";

    // 2nd FIXME: this is not  needed; we should bail earlier...

    // this for loop is needed to handle edge cases where there are not
    // perfectly nested loops TODO: there should not be any imperfectly nested
    // loops
    for (auto en : llvm::enumerate(innerLoops)) {
      _index++;
      auto innerLoop = en.value();
      auto index = en.index();
      llvm::errs() << "[main] Iterating loop: " << index << "\n";

      llvm::errs() << "[main] Collecting args... \n";

      auto arguments = getFuncArgumentsUsedInLoop(innerLoop);

      llvm::errs() << "[main] Collected args: " << arguments.size() << "... \n";

      int currentLoopIndex =
          innerLoop->getAttrOfType<mlir::IntegerAttr>("loop_index").getInt();
      llvm::json::Array args;
      // for every function arguments used in the innermost loop
      for (auto arg : llvm::enumerate(arguments)) {
        llvm::errs() << "[main] Get argument metadata for arg: " << en.index()
                     << "... \n";

        llvm::json::Object argObject =
            getArgumentMetadata(arg.value().getValue(),
                                toInt(arg.value().getKey()), currentLoopIndex);
        args.push_back(std::move(argObject));
      }

      llvm::json::Object indexRange;

      llvm::json::Array locals;
      collectLocals(innerLoop, locals);

      llvm::json::Array constants;
      collectLocalConstants(innerLoop, constants);

      loopObject["constants"] = llvm::json::Value(std::move(constants));
      loopObject["locals"] = llvm::json::Value(std::move(locals));

      indexRange["begin"] = llvm::json::Value(loopNestCounter);
      indexRange["end"] =
          llvm::json::Value(loopNestCounter + loopNestSize - 1 + loopCounter);
      loopObject["loop_position_index_range"] =
          llvm::json::Value(std::move(indexRange));

      loopObject["size"] = llvm::json::Value(loopNestSize);
      loopObject["args"] = llvm::json::Value(std::move(args));
      loopObject["_index"] = llvm::json::Value(_index);
      llvm::errs() << "[main] Collecting loop bounds\n";
      llvm::json::Array bounds;
      llvm::json::Array loopAttrs;
      llvm::json::Array steps;
      collectLoopInfo(innerLoop, bounds, loopAttrs, steps);
      loopObject["bounds"] = llvm::json::Value(std::move(bounds));
      loopObject["steps"] = llvm::json::Value(std::move(steps));

      loopObject["loop_attributes"] = llvm::json::Value(std::move(loopAttrs));

      loops.push_back(llvm::json::Value(std::move(loopObject)));
      loopCounter++;
    }

    loopNestCounter++;

    return mlir::WalkResult::advance();
  });

  if (enableDetailedPrint) {

    llvm::json::Array loopsOrCalls;

    /**
     * Main pass for collecting metadata about calls
     */
    int indexM = 0;
    for (mlir::Operation &op : funcOp.getBody().getOps()) {
      llvm::json::Object loopOrCallObject;
      // TODO:for?
      if (isWhileLoop(&op)) {
        mlir::scf::WhileOp whileOp = cast<mlir::scf::WhileOp>(op);
        loopOrCallObject["type"] = llvm::json::Value("loop_nest");
        loopOrCallObject["outermost_loop_index"] = llvm::json::Value(
            whileOp->getAttrOfType<mlir::IntegerAttr>("loop_index").getInt());
        loopsOrCalls.push_back(llvm::json::Value(std::move(loopOrCallObject)));
      }

      // if call op
      if (isa<mlir::func::CallOp>(op)) {
        mlir::func::CallOp callOp = cast<mlir::func::CallOp>(op);
        llvm::errs() << "[main] Found call: " << indexM << "\n";
        loopOrCallObject["type"] = llvm::json::Value("call");
        loopOrCallObject["callee"] = llvm::json::Value(callOp.getCallee());
        mlir::ValueRange args = callOp.getOperands();
        llvm::json::Array argsArray;
        for (auto arg : llvm::enumerate(args)) {
          llvm::json::Object argObject;

          argObject["name"] = llvm::json::Value(getArgNameFromFuncLocation(
              arg.value().cast<mlir::BlockArgument>()));
          argObject["index"] = llvm::json::Value(
              arg.value().cast<mlir::BlockArgument>().getArgNumber());

          argsArray.push_back(llvm::json::Value(std::move(argObject)));
        }
        loopOrCallObject["args"] = llvm::json::Value(std::move(argsArray));
        loopsOrCalls.push_back(llvm::json::Value(std::move(loopOrCallObject)));
      }

      indexM++;
    }
    function["loops_or_calls"] = llvm::json::Value(std::move(loopsOrCalls));
  }

  llvm::errs() << "[main] Saving result \n";

  function["loops"] = llvm::json::Value(std::move(loops));
  llvm::outs() << formatv("{0:2}", llvm::json::Value(std::move(function)))
               << "\n";

  return 0;
}

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <list>
#include <math.h>
#include <string>

enum IterationOrder {
  PreOrder,
  InOrder,
  PostOrder,
};

class Node {

protected:
  Node *parent_;
  int depth_ = 0;
  Node *left_ = nullptr;
  Node *right_ = nullptr;
  bool isRight_ = false;
  bool isLeft_ = false;
  mlir::Value value_;
  mlir::Operation *ownerLoop = nullptr;

public:
  virtual ~Node() {}
  void setParent(Node *parent) { this->parent_ = parent; }
  Node *getParent() const { return this->parent_; }
  void setDepth(int depth) { this->depth_ = depth; }

  void setValue(mlir::Value value) { this->value_ = value; }
  mlir::Value getValue() { return this->value_; }

  mlir::Operation *getOwnerLoop() { return this->ownerLoop; }
  void setOwnerLoop(mlir::Operation *op) { this->ownerLoop = op; }
  // bool isInductionVar() { this->getOwnerLoop() != nullptr; }

  void setIsRightChild(bool isRight) { this->isRight_ = isRight; }
  bool isRightChild() { return this->isRight_; }
  void setIsLeftChild(bool isLeft) { this->isLeft_ = isLeft; }
  bool isLeftChild() { return this->isLeft_; }

  bool hasRightChild() { return this->right_ != nullptr; }
  bool hasLeftChild() { return this->left_ != nullptr; }

  int getDepth() { return this->depth_; }

  void setLeft(Node *component) {
    this->left_ = component;
    component->setParent(this);
    component->setIsLeftChild(true);
  }

  void setRight(Node *component) {
    this->right_ = component;
    component->setParent(this);
    component->setIsRightChild(true);
  }

  void iterate(std::function<void(Node *)> lambda,
               IterationOrder order = IterationOrder::PreOrder) {
    if (order == IterationOrder::PreOrder) {
      lambda(this);
    }

    if (this->left_) {
      this->left_->iterate(lambda, order);
    }

    if (order == IterationOrder::InOrder) {
      lambda(this);
    }

    if (this->right_) {
      this->right_->iterate(lambda, order);
    }

    if (order == IterationOrder::PostOrder) {
      lambda(this);
    }
  }

  void dump() {
    this->iterate(
        [](Node *node) { printWithDepth(node->getValue(), node->getDepth()); });
  }

  llvm::json::Value toJSON() {

    llvm::json::Array root;
    llvm::json::Array children;

    this->iterate(
        [&](Node *node) {
          llvm::json::Object obj;
          std::string valueStr;
          llvm::raw_string_ostream o(valueStr);

          obj["_treeDepth"] = llvm::json::Value(node->getDepth());
          node->getValue().print(o);
          obj["_value"] = llvm::json::Value(o.str());

          if (node->getOwnerLoop() != nullptr) {
            llvm::raw_string_ostream ownerStream(valueStr);
            // node->getOwnerLoop()->getAttrDictionary().print(ownerStream);

            llvm::json::Object loopDescriptor;

            for (auto attr : node->getOwnerLoop()->getAttrs()) {
              std::string attrValueStr;
              llvm::raw_string_ostream av(attrValueStr);

              attr.getValue().print(av);
              loopDescriptor[attr.getName().str()] =
                  llvm::json::Value(av.str());
            }
            obj["_ownerLoopDescriptor"] =
                llvm::json::Value(std::move(loopDescriptor));
            obj["_parentOp"] = llvm::json::Value(o.str());
          }

          if (node->hasLeftChild() || node->hasRightChild()) {
            obj["children"] = llvm::json::Value(std::move(children));
            children = llvm::json::Array();
          }

          if (node->getParent() != nullptr) {
            children.push_back(llvm::json::Value(std::move(obj)));
          }

          if (node->getParent() == nullptr) {
            root.push_back(llvm::json::Value(std::move(obj)));
          }
        },
        IterationOrder::PostOrder);

    return llvm::json::Value(std::move(root));
  }

  int getTreeDepth() {
    int depth = 0;
    this->iterate([&](Node *node) {
      if (node->getDepth() > depth) {
        depth = node->getDepth();
      }
    });

    return depth;
  }

  int getMultiplyCount() {
    int count = 0;
    this->iterate([&](Node *node) {
      if (!node->getValue().getDefiningOp()) {
        return;
      }

      if (isa<mlir::arith::MulIOp>(node->getValue().getDefiningOp())) {
        count++;
      }
    });

    return count;
  }

  int getMemrefSize() {
    return getMemrefSizeFromMultiplicationCount(this->getMultiplyCount());
  }

  std::string getInductionVarName() {
    assert(this->getOwnerLoop() != nullptr &&
           "Node does not represent an induction variable");

    // TODO: consider start indexing from 0, skip first
    llvm::SmallVector<std::string> inductionVarMap = {"", "z", "y", "x"};

    llvm::errs() << "Owner loop" << this->getOwnerLoop() << "\n";

    auto loopIndex = this->getOwnerLoop()
                         ->getAttr("depth")
                         .cast<mlir::IntegerAttr>()
                         .getInt();

    return inductionVarMap[loopIndex];
  };

  std::string getStencil() {
    llvm::errs() << "Getting stencil for node " << this->getValue() << "\n";
    std::string str = "";

    llvm::StringMap<bool> stencilMap;

    this->iterate(
        [&](Node *node) {
          if (node->getOwnerLoop() != nullptr) {
            auto inductionVariableName = node->getInductionVarName();
            if (str.contains(inductionVariableName))
              str += node->getInductionVarName();
          }
        },
        IterationOrder::PostOrder);

    return str;
  }

  /**
   * @brief Prints an MLIR value with indentation
   * @param value The value to print
   * @param indent The indentation level
   */
  static void printWithDepth(mlir::Value value, int indent) {
    llvm::errs() << std::string(indent, '  ') << value << "\n";
  }

  /**
   * @brief Calculates the rank of a memref from the number of multiplications
   * @param multiplyCount The number of multiplications
   * @return The rank of the memref
   * @details f(m) = (1 + floor(sqrt(1+8m))) / 2
   */
  static int getMemrefSizeFromMultiplicationCount(int multiplications) {
    float sqrt = std::sqrt(1 + 8 * multiplications);
    float result = 1 + sqrt;
    result = result / 2;
    return (int)result;
  }
};
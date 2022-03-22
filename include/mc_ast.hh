#pragma once

#include <memory>
#include <vector>

#include "mc_gencontext.hh"

namespace mlir {
  using DivIOp = SignedDivIOp;
}

namespace mocha {
  enum class ASTType {
    INT32,
    FLOAT32,
  };

  struct ASTNode {
    using Ptr = std::shared_ptr<ASTNode>;

    virtual mlir::Value accept(GenContext &generator) = 0;

    virtual ASTType getType() const = 0;

    virtual ~ASTNode() {}
  };

  struct Cast : public ASTNode {
    ASTNode::Ptr value;
    ASTType from;
    ASTType to;

    Cast(ASTNode::Ptr value, ASTType from, ASTType to)
        : value(value), from(from), to(to) {}

    ASTType getType() const override { return to; }

    mlir::Value accept(GenContext &generator) override {
      if (from == ASTType::INT32)
        return generator.builder.create<mlir::SIToFPOp>(
            generator.locStub(), value->accept(generator),
            generator.builder.getF32Type());
      else
        return generator.builder.create<mlir::FPToSIOp>(
            generator.locStub(), value->accept(generator),
            generator.builder.getI32Type());
    }
  };

  enum class BinOp { Add, Sub, Mul, Div };

  struct BinExpr : public ASTNode {
    ASTNode::Ptr lhs, rhs;
    BinOp op;

    BinExpr(ASTNode::Ptr lhs, BinOp op, ASTNode::Ptr rhs)
        : lhs(lhs), op(op), rhs(rhs) {}

    mlir::Value accept(GenContext &generator) override {
      auto t = lhs->getType();

#define OP_CASE(opname)                                                        \
  case BinOp::opname:                                                          \
    if (t == ASTType::INT32)                                                   \
      return createOp<mlir::opname##IOp>(generator);                           \
    else                                                                       \
      return createOp<mlir::opname##FOp>(generator)

      switch (op) {
        OP_CASE(Add);
        OP_CASE(Sub);
        OP_CASE(Mul);
        OP_CASE(Div);
      }

#undef OP_CASE
    }

    ASTType getType() const override { return lhs->getType(); }

  private:
    template <typename T> mlir::Value createOp(GenContext &generator) {
      return generator.builder.create<T>(
          generator.locStub(), lhs->accept(generator), rhs->accept(generator));
    }
  };

  struct ConstInt : public ASTNode {
    std::int32_t value;

    ConstInt(std::int32_t value) : value(value) {}

    mlir::Value accept(GenContext &generator) override {
      return generator.builder.create<mlir::ConstantOp>(
          generator.locStub(), generator.builder.getI32Type(),
          generator.builder.getI32IntegerAttr(value));
    }

    ASTType getType() const override { return ASTType::INT32; }
  };

  struct ConstFloat : public ASTNode {
    float value;

    ConstFloat(float value) : value(value) {}

    mlir::Value accept(GenContext &generator) override {
      return generator.builder.create<mlir::ConstantOp>(
          generator.locStub(), generator.builder.getF32Type(),
          generator.builder.getF32FloatAttr(value));
    }

    ASTType getType() const override { return ASTType::INT32; }
  };

  struct Load : public ASTNode {
    int loadingField;
    ASTType dtype;
    std::vector<ASTNode::Ptr> offsets;

    Load(int loadingField, ASTType dtype,
         const std::vector<ASTNode::Ptr> offsets)
        : loadingField(loadingField), dtype(dtype), offsets(offsets) {}

    mlir::Value accept(GenContext &generator) override {
      std::vector<mlir::Value> offsetValue;
      for (auto optr : offsets)
        offsetValue.push_back(optr->accept(generator));
      return generator.builder.create<mlir::AffineLoadOp>(
          generator.locStub(), generator.getField(loadingField), offsetValue);
    }

    ASTType getType() const override { return dtype; }
  };

  struct Store : public ASTNode {
    int storingField;
    std::vector<ASTNode::Ptr> offsets;
    ASTNode::Ptr value;

    Store(ASTNode::Ptr value, int storingField,
          const std::vector<ASTNode::Ptr> &offsets)
        : storingField(storingField), offsets(offsets), value(value) {}

    mlir::Value accept(GenContext &generator) override {
      std::vector<mlir::Value> offsetValue;
      for (auto optr : offsets)
        offsetValue.push_back(optr->accept(generator));
      generator.builder.create<mlir::AffineStoreOp>(
          generator.locStub(), value->accept(generator),
          generator.getField(storingField), offsetValue);
      return generator.builder.create<mlir::ConstantOp>(
          generator.locStub(), generator.builder.getI32Type(),
          generator.builder.getI32IntegerAttr(0));
    }

    ASTType getType() const override { return ASTType::INT32; }
  };

  struct Indexer : public ASTNode {
    int position;

    Indexer(int position) : position(position) {}

    mlir::Value accept(GenContext &generator) override {
      return generator.getIndexer(position);
    }

    ASTType getType() const override { return ASTType::INT32; }
  };
} // namespace mocha
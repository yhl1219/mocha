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
  };

  struct Field : public ASTNode {
    using Ptr = std::shared_ptr<Field>;

    ASTType dtype;
    int id;
    std::vector<int> shape;

    Field(ASTType dtype, int id, const std::vector<int> &shape)
        : dtype(dtype), shape(shape), id(id) {}

    // only stub method
    mlir::Value accept(GenContext &generator) override { return mlir::Value(); }

    // only stub method
    ASTType getType() const override { return ASTType::INT32; }
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
            generator.locStub(), value->accept(generator));
      else
        return generator.builder.create<mlir::FPToSIOp>(
            generator.locStub(), value->accept(generator));
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
      auto type = generator.builder.getI32Type();
      return generator.builder.create<mlir::ConstantIntOp>(generator.locStub(),
                                                           value, type);
    }

    ASTType getType() const override { return ASTType::INT32; }
  };

  struct ConstFloat : public ASTNode {
    float value;

    ConstFloat(float value) : value(value) {}

    mlir::Value accept(GenContext &generator) override {
      auto type = generator.builder.getF32Type();
      return generator.builder.create<mlir::ConstantFloatOp>(
          generator.locStub(), value, type);
    }

    ASTType getType() const override { return ASTType::INT32; }
  };

  struct Load : public ASTNode {
    Field::Ptr loadingField;
    std::vector<ASTNode::Ptr> offsets;

    Load(Field::Ptr loadingField, const std::vector<ASTNode::Ptr> offsets)
        : loadingField(loadingField), offsets(offsets) {}

    mlir::Value accept(GenContext &generator) override {
      std::vector<mlir::Value> offsetValue;
      for (auto optr : offsets)
        offsetValue.push_back(optr->accept(generator));
      return generator.builder.create<mlir::AffineLoadOp>(
          generator.locStub(), generator.getField(loadingField->id),
          offsetValue);
    }

    ASTType getType() const override { return loadingField->dtype; }
  };

  struct Store : public ASTNode {
    Field::Ptr storingField;
    std::vector<ASTNode::Ptr> offsets;
    ASTNode::Ptr value;

    Store(Field::Ptr storingField, const std::vector<ASTNode::Ptr> &offsets,
          ASTNode::Ptr value)
        : storingField(storingField), offsets(offsets), value(value) {}

    mlir::Value accept(GenContext &generator) override {
      std::vector<mlir::Value> offsetValue;
      for (auto optr : offsets)
        offsetValue.push_back(optr->accept(generator));
      generator.builder.create<mlir::AffineStoreOp>(
          generator.locStub(), value->accept(generator),
          generator.getField(storingField->id), offsets);
      return generator.builder.create<mlir::ConstantIntOp>(generator.locStub(),
                                                           0);
    }

    ASTType getType() const override { return storingField->dtype; }
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
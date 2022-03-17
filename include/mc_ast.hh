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

  struct Field {
    using Ptr = std::shared_ptr<Field>;

    ASTType dtype;
    std::vector<int> shape;

    Field(ASTType dtype, const std::vector<int> &shape)
        : dtype(dtype), shape(shape) {}
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

    ASTType getType() const override {
      auto lType = lhs->getType();
      auto rType = rhs->getType();
      if (lType != rType) {
        throw std::exception("type mismatch"); // should not happen
      } else {
        return lType;
      }
    }

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

  struct Load : public ASTNode {};

  struct Indexer : public ASTNode {
    int position;

    Indexer(int position) : position(position) {}

    mlir::Value accept(GenContext &generator) override {
      return generator.getIndexer(position);
    }

    ASTType getType() const override { return ASTType::INT32; }
  };
} // namespace mocha
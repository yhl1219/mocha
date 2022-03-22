#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

namespace mocha {
  class GenContext {
  public:
    GenContext(mlir::MLIRContext &context)
        : _context(context), builder(&context) {}

    mlir::MLIRContext &_context;
    mlir::ModuleOp module;
    mlir::OpBuilder builder;
    std::vector<mlir::Value> indexer;
    std::vector<mlir::Value> fields;

    // create loc stub, not the real one
    mlir::Location locStub() {
      return mlir::FileLineColLoc::get("mocha", 0, 0, &_context);
    }

    mlir::Value getIndexer(int position) { return indexer[position]; }

    mlir::Value getField(int id) { return fields[id]; }
  };
} // namespace mocha
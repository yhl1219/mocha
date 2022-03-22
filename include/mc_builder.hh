#pragma once

#include "mc_ast.hh"
#include "mc_gencontext.hh"

#include <iostream>

#include <json/json.h>

namespace mocha {
  class MochaGenerator {
    std::unique_ptr<mlir::MLIRContext> context;
    std::unique_ptr<GenContext> gen;

    void parse_string(const std::string &str, Json::Value &v) {
      JSONCPP_STRING err;
      auto readerBuilder = Json::CharReaderBuilder();
      auto reader = readerBuilder.newCharReader();
      reader->parse(str.c_str(), str.c_str() + str.length(), &v, &err);
    }

    mlir::MemRefType getFieldType(int dtype,
                                        const std::vector<int64_t> &shape) {
      if (dtype == 0)
        return mlir::MemRefType::get(shape, gen->builder.getI32Type());
      else
        return mlir::MemRefType::get(shape, gen->builder.getF32Type());
    }

    std::vector<int64_t> getShape(const Json::Value &v) {
      std::vector<int64_t> shape;
      for (int i = 0; i < (int)v.size(); ++i)
        shape.push_back(v[i].asInt());
      return shape;
    }

    ASTNode::Ptr rebuild_ast(const Json::Value &v) {
      auto type = v["type"].asString();
      if (type == "indexer") {
        auto position = v["position"].asInt();
        return std::make_shared<Indexer>(position);
      } else if (type == "store") {
        auto value = rebuild_ast(v["value"]);
        auto field = v["field"].asInt();

        // get offset
        auto off_json = v["offsets"];
        std::vector<ASTNode::Ptr> offsets;
        for (int i = 0; i < (int)off_json.size(); ++i)
          offsets.push_back(rebuild_ast(off_json[i]));

        return std::make_shared<Store>(value, field, offsets);
      } else if (type == "load") {
        auto field = v["field"].asInt();
        auto dtype = (ASTType)(v["dtype"].asInt());

        // get offset
        auto off_json = v["offsets"];
        std::vector<ASTNode::Ptr> offsets;
        for (int i = 0; i < (int)off_json.size(); ++i)
          offsets.push_back(rebuild_ast(off_json[i]));
        return std::make_shared<Load>(field, dtype, offsets);
      } else if (type == "const") {
        auto dtype = (ASTType)(v["dtype"].asInt());
        if (dtype == ASTType::INT32)
          return std::make_shared<ConstInt>(v["value"].asInt());
        else
          return std::make_shared<ConstFloat>(v["value"].asFloat());
      } else if (type == "cast") {
        auto value = rebuild_ast(v["value"]);
        auto from = (ASTType)(v["from"].asInt());
        auto to = (ASTType)(v["to"].asInt());
        return std::make_shared<Cast>(value, from, to);
      } else if (type == "binexpr") {
        auto left = rebuild_ast(v["left"]);
        auto right = rebuild_ast(v["right"]);
        auto op = (BinOp)(v["op"].asInt());
        return std::make_shared<BinExpr>(left, op, right);
      } else {
        throw std::exception();
      }
    }

  public:
    MochaGenerator(const std::string &jsonFieldInfo,
                   const std::string &jsonASTInfo) {
      context = std::make_unique<mlir::MLIRContext>();
      context->getOrLoadDialect<mlir::AffineDialect>();
      context->getOrLoadDialect<mlir::StandardOpsDialect>();
      gen = std::make_unique<GenContext>(*context);

      auto module = mlir::ModuleOp::create(gen->locStub());
      gen->builder.setInsertionPointToStart(module.getBody());

      // handle function definition, register fields as parameters
      {
        Json::Value field_info, ast_json;
        parse_string(jsonFieldInfo, field_info);

        // register all fields and get the signature
        std::vector<mlir::Type> signature;

        // first, handle the main field
        auto main_field = field_info["main"];
        auto main_shape = getShape(main_field["shape"]);
        signature.push_back(
            getFieldType(main_field["dtype"].asInt(), main_shape));

        // then the rest
        auto rest_fields = field_info["rest"];
        for (int i = 0; i < (int)rest_fields.size(); ++i)
          signature.push_back(getFieldType(rest_fields[i]["dtype"].asInt(),
                                           getShape(rest_fields[i]["shape"])));

        // the the function signature
        auto funcType = gen->builder.getFunctionType(
            signature, std::vector<mlir::Type>{gen->builder.getI32Type()});

        // dump it, just for debug
        printf("[ mocha_mlir ] compiled signature is ");
        funcType.dump();
        printf("\n");

        // figure out the funcion block, next, we are going to build the
        // function body
        auto func = gen->builder.create<mlir::FuncOp>(gen->locStub(), "mocha",
                                                      funcType);

        {
          // generate function body, set the position to the start of the entry
          // block
          printf("[ mocha_mlir ] start to emit function body\n");
          auto entryBlock = func.addEntryBlock();
          gen->builder.setInsertionPointToStart(entryBlock);
          auto arguments = entryBlock->getArguments();
          for (auto &arg : arguments)
            gen->fields.push_back(arg);

          printf("[ mocha_mlir ] start to rebuild ast in cpp\n");
          parse_string(jsonASTInfo, ast_json);
          auto ast = rebuild_ast(ast_json);

          printf("[ mocha_mlir ] start to emit affine loop\n");
          std::vector<int64_t> begin, steps;
          for (size_t i = 0; i < main_shape.size(); ++i)
            begin.push_back(0), steps.push_back(1);
          mlir::buildAffineLoopNest(
              gen->builder, gen->locStub(), begin, main_shape, steps,
              [&](mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::ValueRange range) {
                for (auto i : range)
                  gen->indexer.push_back(i);
                ast->accept(*gen);
              });

          // ast->accept(*gen);
        }

        // dump the module
        module->dump();
      }
    }
  };
} // namespace mocha
from array import array
import ast
import inspect
from typing import cast

from mocha.exception import MochaSyntaxException, MochaTypeException
from mocha.ir import MochaAstVistor, MochaSignature


def __getsrcfile(func):
    f = inspect.getsourcefile(func)
    return '<unknown>' if f is None else f


def __syntax_exception(func, message: str):
    f = __getsrcfile(func)
    context, line = inspect.getsourcelines(func)
    raise MochaSyntaxException(line, context[line], f, message)


def __extract_signature(func, is_method: bool):
    # function that filter out unsupported grammar
    def __validate_parameter(param: inspect.Parameter):
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            __syntax_exception(
                func, f'parameter kind {param.kind} is not supported')
        if param.default != inspect.Parameter.empty:
            __syntax_exception(
                func, f'default value for parameter is not supported')

    signature = inspect.signature(func)

    # return type
    return_type = None if signature.return_annotation in (
        inspect._empty, None) else signature.return_annotation
    if return_type not in (int, float, bool, array, None):
        __syntax_exception(func, f'invalid return type {return_type}')

    # params
    extracted_signature = []
    for i, (_, param) in enumerate(signature.parameters.items()):
        __validate_parameter(param)
        param_name, param_type = param.name, param.annotation

        # the parameter self
        if param_type is inspect.Parameter.empty:
            if i == 0 and is_method:
                # TODO: support method as compiled
                param_type = None  # None represents is reserved
            else:
                __syntax_exception(
                    func, f'parameter requires type')

        # make sure param_type is valid and supported
        if param_type not in (int, float, bool, array):
            __syntax_exception(func, f'invalid type {param_type}')

        extracted_signature.append((param_name, param_type))

    return MochaSignature(return_type, extracted_signature)


class CompileOption:
    def __init__(self, is_method: bool) -> None:
        self.is_method = is_method


def __default_option() -> CompileOption:
    return CompileOption(is_method=False)


def __ir_transform(func, options: CompileOption):
    signature = __extract_signature(func, options.is_method)

    func_module = ast.parse(inspect.getsource(
        func), __getsrcfile(func), type_comments=True)

    assert type(func_module) == ast.Module and type(
        func_module.body[0]) == ast.FunctionDef

    func_def = cast(ast.FunctionDef, func_module.body[0])

    for stat in func_def.body:
        print(ast.dump(stat))

    visitor = MochaAstVistor(func, __getsrcfile(func), signature)
    ir = visitor.visit_FunctionDef(func_def)

    return lambda x: 2


def compiled(options: CompileOption = __default_option()):
    def compiled_decorator(func):
        if not inspect.isfunction(func):
            raise MochaTypeException(function, type(func))
        return __ir_transform(func, options)
    return compiled_decorator

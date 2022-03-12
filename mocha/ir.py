import ast
from fileinput import filename
import inspect
from typing import Any

from mocha.exception import MochaSyntaxException


class Matrix:
    # TODO: implement stub for matrix
    def __init__(self, shape: tuple[int, int]) -> None:
        self.m, self.n = shape

    def __setitem__(self, location: tuple[int, int], value: Any):
        pass

    def __getitem__(self, location: tuple[int, int], value: Any):
        pass

# TODO: implement stub for vector


class Vector:
    def __init__(self) -> None:
        pass

    def __setitem__(self, location: int, value: Any):
        pass

    def __getitem__(self, location: int, value: Any):
        pass


vector = Vector
matrix = Matrix


class MochaSignature:
    def __init__(self, return_type: Any, arguments: list[tuple[str, Any]]) -> None:
        self.return_type = return_type
        self.arguments = arguments
        self.__build_argument_table()

    def __build_argument_table(self):
        self.args_table = {}
        for name, type in self.arguments:
            self.args_table[name] = type


class MochaIR:
    pass


class MochaVar:
    def __init__(self, name: str, id: int, type: type) -> None:
        self.name = name
        self.id = id
        self.type = type


class MochaExpression(MochaIR):
    pass


class MochaConstExpression(MochaExpression):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class MochaUnaryExpression(MochaExpression):
    def __init__(self, op: str, rhs: MochaExpression) -> None:
        super().__init__()
        self.op, self.rhs = op, rhs


class MochaBinaryExpression(MochaExpression):
    def __init__(self, lhs: MochaExpression, op: str, rhs: MochaExpression) -> None:
        super().__init__()
        self.lhs, self.op, self.rhs = lhs, op, rhs


class MochaAssign(MochaIR):
    def __init__(self, target: MochaVar, value: MochaExpression) -> None:
        super().__init__()
        self.target = target
        self.value = value


class MochaLoad(MochaIR):
    pass


class MochaAstVistor(ast.NodeVisitor):
    def __init__(self, func, filename: str, signature: MochaSignature) -> None:
        super().__init__()
        self.signature = signature

        # generate vars from parameters
        self.var_id = 0
        self.var_table = {}
        self.__initialize_parameters()

        # generate context to report something wrong
        self.context, self.start_lineno = inspect.getsourcelines(func)
        self.filename = filename

        self.building_sequence = []

    def __new_var(self, type: type, node: ast.AST | None = None, name: str | None = None):
        mangle_name = f'_{self.var_id}' if name is None else f'_{name}'
        if mangle_name in self.var_table:
            if node is None:
                raise MochaSyntaxException(
                    self.start_lineno, self.context[self.start_lineno], self.filename, f'duplicate var {name}')
            else:
                self.__syntax_exception(node, f'duplicate var {name}')

        # generate var
        var = MochaVar(mangle_name, self.var_id, type)

        self.var_id += 1
        self.var_table[name] = var

        return var

    def __initialize_parameters(self):
        for par_name, par_type in self.signature.arguments:
            self.__new_var(par_type, None, par_name)

    def __syntax_exception(self, node: ast.AST, message: str):
        raise MochaSyntaxException(
            node.lineno, self.context[self.start_lineno - node.lineno], self.filename, message)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        for stat in node.body:
            self.visit(stat)

    def __find_var(self, node: ast.AST, name: str) -> MochaVar:
        if name not in self.var_table:
            self.__syntax_exception(node, f'undefined reference to {name}')
        return self.var_table[name]

    def visit_Assign(self, node: ast.Assign) -> None:
        # extract assign targets
        assign_targets = node.targets
        assign_targets.reverse()

        # generate each assignment to an ir
        for target in assign_targets:
            if isinstance(target, ast.Name):
                # simple assignment
                target_var = self.__find_var(node, target.id)
                self.building_sequence.append(
                    MochaAssign(target_var, self.visit(node.value)))
            elif isinstance(target, ast.Subscript):
                # load
                pass

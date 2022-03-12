import ast
import inspect
from typing import Any

from mocha.exception import MochaSyntaxException


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
        self.var_table[mangle_name] = var

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
        mangle_name = f'_{self.var_id}' if name is None else f'_{name}'
        if mangle_name not in self.var_table:
            self.__syntax_exception(node, f'undefined reference to {name}')
        return self.var_table[mangle_name]

    def visit_Constant(self, node: ast.Constant) -> MochaConstExpression:
        return MochaConstExpression(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # filter out op
        ops: dict[type, str] = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Is: '==',
            ast.IsNot: '!=',
            ast.MatMult: '@',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Lt: '<',
            ast.LtE: '<='
        }

        try:
            bop = ops[type(node.op)]
        except KeyError:
            self.__syntax_exception(node, f'{str(node.op)} is not supported')
        return MochaBinaryExpression(self.visit(node.left), bop, self.visit(node.right))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        ops: dict[type, str] = {
            ast.UAdd: '+',
            ast.USub: '-',
            ast.Not: '!',
        }

        try:
            uop = ops[type(node.op)]
        except KeyError:
            self.__syntax_exception(node, f'{str(node.op)} is not supported')
        return MochaUnaryExpression(uop, self.visit(node.operand))

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
                # TODO: implement load
                pass
            else:
                self.__syntax_exception(
                    node, f'parallel assignment is unsupported')

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        print(str(node.annotation))
        target = node.target

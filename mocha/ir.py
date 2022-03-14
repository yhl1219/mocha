import ast
from functools import reduce
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
    def dump(self, indent: int = 0) -> None:
        print(' ' * indent + self.__class__.__name__ + ':')
        for name, value in self.__dict__.items():
            if isinstance(value, MochaIR):
                print(' ' * indent + f'  {name} =')
                value.dump(indent + 6)
            elif isinstance(value, list):
                print(' ' * indent + f'  {name} = <list>')
                for i in value:
                    if isinstance(i, MochaIR):
                        i.dump(indent + 6)
            else:
                print(' ' * indent + f'  {name} = {value}')


class MochaVar:
    def __init__(self, name: str, id: int, type: type) -> None:
        self.name = name
        self.id = id
        self.type = type

    def __str__(self) -> str:
        return f'({self.id}){self.name} :: {self.type}'


class MochaExpression(MochaIR):
    pass


class MochaConstExpression(MochaExpression):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class MochaVarExpression(MochaExpression):
    def __init__(self, var: MochaVar) -> None:
        super().__init__()
        self.var = var


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


class MochaBlock(MochaIR):
    def __init__(self, irs: list[MochaIR]) -> None:
        super().__init__()
        self.irs = irs


class MochaIf(MochaIR):
    def __init__(self, condition: MochaExpression, if_branch: MochaIR, else_branch: MochaIR) -> None:
        super().__init__()
        self.condition = condition
        self.if_branch = if_branch
        self.else_branch = else_branch


class MochaLoad(MochaIR):
    pass


class MochaNop(MochaIR):
    pass


class MochaAstVistor(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> Any:
        self.__syntax_exception(
            node, f'{node.__class__.__name__} is not supported')

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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> MochaIR:
        return MochaBlock([self.visit(stat) for stat in node.body])

    def __find_var(self, node: ast.AST, name: str) -> MochaVar:
        mangle_name = f'_{self.var_id}' if name is None else f'_{name}'
        if mangle_name not in self.var_table:
            self.__syntax_exception(node, f'undefined reference to {name}')
        return self.var_table[mangle_name]

    def visit_Constant(self, node: ast.Constant) -> MochaConstExpression:
        return MochaConstExpression(node.value)

    def visit_Name(self, node: ast.Name) -> Any:
        return MochaVarExpression(self.__find_var(node, node.id))

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        # filter out op
        ops: dict[type, str] = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
        }

        try:
            bop = ops[type(node.op)]
        except KeyError:
            self.__syntax_exception(node, f'{str(node.op)} is not supported')
        return MochaBinaryExpression(self.visit(node.left), bop, self.visit(node.right))

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        ops: dict[type, str] = {
            ast.And: '&&',
            ast.Or: '||',
        }

        try:
            bop = ops[type(node.op)]
        except KeyError:
            self.__syntax_exception(node, f'{str(node.op)} is not supported')

        vexprs = map(self.visit, node.values)
        return reduce(lambda a, b: MochaBinaryExpression(a, bop, b), vexprs)

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

    def visit_Compare(self, node: ast.Compare) -> Any:
        if len(node.ops) != 1:
            self.__syntax_exception(
                node, 'multiple comparison is not supported')

        ops: dict[type, str] = {
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Eq: '==',
            ast.NotEq: '!='
        }

        try:
            bop = ops[type(node.ops[0])]
        except KeyError:
            self.__syntax_exception(
                node, f'{str(node.ops[0])} is not supported')
        return MochaBinaryExpression(self.visit(node.left), bop, self.visit(node.comparators[0]))

    def visit_Assign(self, node: ast.Assign) -> MochaIR:
        # extract assign targets
        assign_targets = node.targets
        assign_targets.reverse()

        # generate each assignment to an ir
        irs = []
        for target in assign_targets:
            if isinstance(target, ast.Name):
                # simple assignment
                target_var = self.__find_var(node, target.id)
                irs.append(
                    MochaAssign(target_var, self.visit(node.value)))
            elif isinstance(target, ast.Subscript):
                # TODO: implement store
                pass
            else:
                self.__syntax_exception(
                    node, f'parallel assignment is unsupported')
        return MochaBlock(irs)

    def __resolve_annotation(self, annotation: ast.expr) -> Any:
        pass

    def visit_AnnAssign(self, node: ast.AnnAssign) -> MochaIR:
        type_annotation = self.__resolve_annotation(node.annotation)

        node_target = node.target
        assert isinstance(node_target, ast.Name)

        target_var = self.__new_var(type_annotation, node, node_target.id)

        # if there's assignment
        if(node.value is not None):
            return MochaAssign(target_var, self.visit(node.value))
        else:
            return MochaNop()

    def visit_Expr(self, node: ast.Expr) -> None:
        # should I do something, should I ?
        pass

    def visit_If(self, node: ast.If) -> Any:
        condition = self.visit(node.test)
        return MochaIf(condition, None, [])

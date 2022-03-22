from dataclasses import fields
import json
from typing import Any, Callable, cast
import numpy as np


class IRNode:
    pass


class Expr(IRNode):
    def __add__(self, rhs: Any):
        if isinstance(rhs, Expr):
            return BinExpr(self, '+', rhs)
        elif type(rhs) in (int, float):
            return BinExpr(self, '+', Const(rhs))
        else:
            raise TypeError(f'no operator add between Expr and {type(rhs)}')

    def __sub__(self, rhs: Any):
        if isinstance(rhs, Expr):
            return BinExpr(self, '-', rhs)
        elif type(rhs) in (int, float):
            return BinExpr(self, '-', Const(rhs))
        else:
            raise TypeError(f'no operator add between Expr and {type(rhs)}')

    def __mul__(self, rhs: Any):
        if isinstance(rhs, Expr):
            return BinExpr(self, '+', rhs)
        elif type(rhs) in (int, float):
            return BinExpr(self, '+', Const(rhs))
        else:
            raise TypeError(f'no operator add between Expr and {type(rhs)}')

    def __div__(self, rhs: Any):
        if isinstance(rhs, Expr):
            return BinExpr(self, '/', rhs)
        elif type(rhs) in (int, float):
            return BinExpr(self, '/', Const(rhs))
        else:
            raise TypeError(f'no operator add between Expr and {type(rhs)}')

    def __radd__(self, lhs: Any):
        if isinstance(lhs, Expr):
            return BinExpr(lhs, '+', self)
        elif type(lhs) in (int, float):
            return BinExpr(Const(lhs), '+', self)
        else:
            raise TypeError(f'no operator add between Expr and {type(lhs)}')

    def __rsub__(self, lhs: Any):
        if isinstance(lhs, Expr):
            return BinExpr(lhs, '+', self)
        elif type(lhs) in (int, float):
            return BinExpr(Const(lhs), '+', self)
        else:
            raise TypeError(f'no operator add between Expr and {type(lhs)}')

    def __rmul__(self, lhs: Any):
        if isinstance(lhs, Expr):
            return BinExpr(lhs, '+', self)
        elif type(lhs) in (int, float):
            return BinExpr(Const(lhs), '+', self)
        else:
            raise TypeError(f'no operator add between Expr and {type(lhs)}')

    def __rdiv__(self, lhs: Any):
        if isinstance(lhs, Expr):
            return BinExpr(lhs, '+', self)
        elif type(lhs) in (int, float):
            return BinExpr(Const(lhs), '+', self)
        else:
            raise TypeError(f'no operator add between Expr and {type(lhs)}')

    def type_check(self):
        pass


class Field:
    BUILD_IR: bool = False
    BUILT_IR = None

    def __init__(self, dtype, shape: tuple[int, ...]) -> None:
        self.dtype = dtype
        self.__shape = shape

        # perform some check to dtype
        if dtype not in (int, float):
            raise TypeError(f'invalid type {dtype}')

        # perform some check to the shape
        if len(shape) > 3:
            raise TypeError('dimension greater than 3 is not supported')

        total_size = 1
        for i in range(0, len(shape)):
            ndim_size = shape[i]
            if ndim_size < 0 or type(ndim_size) != int:
                raise TypeError(f'invalid shape {ndim_size} in dimension {i}')
            total_size *= ndim_size
        self.__size = total_size

        # just ...
        self.data = np.zeros(
            shape, dtype=np.int32 if dtype == int else np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.__shape

    @property
    def dim(self) -> int:
        return len(self.__shape)

    @property
    def size(self) -> int:
        return self.__size

    def __len__(self) -> int:
        return self.__size

    def __fix_term(self, t):
        if isinstance(t, Expr):
            return t
        elif type(t) in (int, float):
            return Const(t)
        else:
            raise TypeError(f'invalid type {type(t)}')

    def __getitem__(self, key):
        if Field.BUILD_IR:
            offset = []
            if isinstance(key, tuple):
                for t in key:
                    offset.append(self.__fix_term(t))
            else:
                offset.append(self.__fix_term(key))
            return Load(self, offset)
        else:
            return self.data[key]

    def __setitem__(self, key, value):
        if Field.BUILD_IR:
            offset = []
            if isinstance(key, tuple):
                for t in key:
                    offset.append(self.__fix_term(t))
            else:
                offset.append(self.__fix_term(key))
            Field.BUILT_IR = Store(self, offset, self.__fix_term(value))
        else:
            self.data[key] = value


class Cast(Expr):
    def __init__(self, v: Expr, origin: type, to: type) -> None:
        super().__init__()
        self.v, self.origin, self.to = v, origin, to

    def type_check(self):
        return self.to


class BinExpr(Expr):
    def __init__(self, lhs: Expr, op: str, rhs: Expr) -> None:
        super().__init__()
        self.lhs, self.op, self.rhs = lhs, op, rhs

    def type_check(self):
        ltype = self.lhs.type_check()
        rtype = self.rhs.type_check()
        if ltype == rtype:
            return ltype
        else:
            if ltype == int:
                self.lhs = Cast(self.lhs, int, float)
            else:
                self.rhs = Cast(self.rhs, int, float)
            return float


class Const(Expr):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = value

    def type_check(self):
        return type(self.value)


class Load(Expr):
    def __init__(self, field: Field, offset: list[Expr]) -> None:
        super().__init__()
        self.field, self.offset = field, offset

    def type_check(self):
        if len(self.offset) != self.field.dim:
            raise TypeError('dimension of offset mismatch dimension of field')
        for o in self.offset:
            otype = o.type_check()
            if otype != int:
                raise TypeError('invalid type float in index')
        return self.field.dtype


class Store(Expr):
    def __init__(self, field: Field, offset: list[Expr], value: Expr) -> None:
        super().__init__()
        self.field, self.offset, self.value = field, offset, value

    def type_check(self):
        if len(self.offset) != self.field.dim:
            raise TypeError('dimension of offset mismatch dimension of field')
        for o in self.offset:
            otype = o.type_check()
            if otype != int:
                raise TypeError('invalid type float in index')
        vtype = self.value.type_check()
        if vtype != self.field.dtype:
            if self.field.dtype == float:
                self.value = Cast(self.value, int, float)
            else:
                self.value = Cast(self.value, float, int)


class Indexer(Expr):
    def __init__(self, position: int) -> None:
        super().__init__()
        self.position = position

    def type_check(self):
        return int


class Coord3D:
    def __init__(self) -> None:
        self.x = Indexer(0)
        self.y = Indexer(1)
        self.z = Indexer(2)


class NodeVisitor:
    def __init__(self) -> None:
        self.main_field: Field = cast(Field, None)
        self.fields: list[Field] = []

    def visit(self, node: IRNode):
        if isinstance(node, BinExpr):
            return self.visit_binexpr(node)
        elif isinstance(node, Cast):
            return self.visit_cast(node)
        elif isinstance(node, Const):
            return self.visit_const(node)
        elif isinstance(node, Load):
            return self.visit_load(node)
        elif isinstance(node, Store):
            return self.visit_store(node)
        elif isinstance(node, Indexer):
            return self.visit_indexer(node)

    def visit_indexer(self, node: Indexer):
        return {
            "type": "indexer",
            "position": node.position
        }

    def visit_field(self, f: Field, is_main: bool = False):
        if is_main:
            self.main_field = f
            id = 0
        else:
            if f in self.fields:
                id = self.fields.index(f) + 1
            else:
                self.fields.append(f)
                id = len(self.fields)
        return id

    def visit_store(self, node: Store):
        self.shape = node.field.shape
        return {
            "type": "store",
            "value": self.visit(node.value),
            "field": self.visit_field(node.field, True),
            "offsets": list(map(self.visit, node.offset))
        }

    def visit_load(self, node: Load):
        return {
            "type": "load",
            "field": self.visit_field(node.field),
            "dtype": self.format_type(node.field.dtype),
            "offsets": list(map(self.visit, node.offset))
        }

    def visit_const(self, node: Const):
        return {
            "type": "const",
            "dtype": self.format_type(type(node.value)),
            "value": node.value
        }

    def format_type(self, t: type):
        return 0 if t == int else 1

    def visit_cast(self, node: Cast):
        return {
            "type": "cast",
            "value": self.visit(node.v),
            "from": self.format_type(node.origin),
            "to": self.format_type(node.to)
        }

    def visit_binexpr(self, node: BinExpr):
        return {
            "type": "binexpr",
            "left": self.visit(node.lhs),
            "op": {"+": 0, "-": 1, "*": 2, "/": 3}[node.op],
            "right": self.visit(node.rhs)
        }

    def export_fields(self) -> dict:
        a = {"main": {
            "dtype": self.format_type(self.main_field.dtype), "shape": self.main_field.shape}, "rest": []}
        for f in self.fields:
            a['rest'].append(
                {"dtype": self.format_type(f.dtype), "shape": f.shape})
        return a


def stencil(callable: Callable[[Coord3D], None]):
    Field.BUILD_IR = True
    callable(Coord3D())
    Field.BUILD_IR = False

    store = Field.BUILT_IR
    if store is None:
        raise Exception('invalid stencil expression!')

    # perform type check and field check
    store.type_check()

    visitor = NodeVisitor()
    formatted_code = visitor.visit(store)

    import build.mocha_mlir as mocha_mlir
    generator = mocha_mlir.MochaGenerator(json.dumps(
        visitor.export_fields()), json.dumps(formatted_code))

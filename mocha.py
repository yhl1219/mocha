from typing import Any, Callable
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
    def __init__(self, position: str) -> None:
        super().__init__()
        self.position = position

    def type_check(self):
        return int


class Coord3D:
    def __init__(self) -> None:
        self.x = Indexer('x')
        self.y = Indexer('y')
        self.z = Indexer('z')


class NodeVisitor:
    def visit(self, node: IRNode):
        if isinstance(node, BinExpr):
            self.visit_binexpr(node)
        elif isinstance(node, Cast):
            self.visit_cast(node)
        elif isinstance(node, Const):
            self.visit_const(node)
        elif isinstance(node, Load):
            self.visit_load(node)
        elif isinstance(node, Store):
            self.visit_store(node)
        elif isinstance(node, Indexer):
            self.visit_indexer(node)

    def visit_indexer(self, node: Indexer):
        pass

    def visit_store(self, node: Store):
        pass

    def visit_load(self, node: Load):
        pass

    def visit_const(self, node: Const):
        pass

    def visit_cast(self, node: Cast):
        pass

    def visit_binexpr(self, node: BinExpr):
        pass


def stencil(callable: Callable[[Coord3D], None]):
    Field.BUILD_IR = True
    callable(Coord3D())
    Field.BUILD_IR = False

    store = Field.BUILT_IR
    if store is None:
        raise Exception('invalid stencil expression!')

    # perform type check and field check
    store.type_check()

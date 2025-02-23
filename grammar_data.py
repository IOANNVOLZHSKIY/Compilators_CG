import abc
import sys
sys.path.append('..')

import parser_edsl_new as pe
from lexer import *
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class Node(abc.ABC):
    @abc.abstractmethod
    def check(self, symbols: dict):
        pass

class SemanticError(pe.Error):
    def __init__(self, pos, message):
        self.pos = pos
        self.__message = message

    @property
    def message(self):
        return self.__message

    def __str__(self):
        return f"Semantic error at {self.pos}: {self.__message}"

class UndefinedError(SemanticError):
    pass

class TypeMismatchError(SemanticError):
    pass

class DuplicateDeclarationError(SemanticError):
    pass

class InvalidOperationError(SemanticError):
    pass

@dataclass
class Program(Node):
    top_levels: List[Node] = field(default_factory=list)

    def check(self, symbols: dict = None):
        if symbols is None:
            symbols = {}
        for decl in self.top_levels:
            decl.check(symbols)

@dataclass
class TypeDecl(Node):
    name: str
    spec: Node

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Type '{self.name}' already declared")
        symbols[self.name] = self.spec
        self.spec.check(symbols)

@dataclass
class StructSpec(Node):
    fields: List["StructField"] = field(default_factory=list)

    def check(self, symbols: dict):
        seen = set()
        for field in self.fields:
            if field.name in seen:
                raise DuplicateDeclarationError(field.pos, f"Duplicate field '{field.name}'")
            seen.add(field.name)
            field.check(symbols)

@dataclass
class StructField(Node):
    name: str
    fieldType: Node

    def check(self, symbols: dict):
        self.fieldType.check(symbols)

@dataclass
class ClassSpec(Node):
    bases: List[str] = field(default_factory=list)
    members: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        for member in self.members:
            member.check(symbols)

@dataclass
class ClassMember(Node):
    name: str
    memberType: Optional[Node] = None  # если поле
    methodDecl: Optional[Node] = None  # если метод

    def check(self, symbols: dict):
        if self.memberType:
            self.memberType.check(symbols)
        if self.methodDecl:
            self.methodDecl.check(symbols)

@dataclass
class ClassField(Node):
    name: str
    fieldType: Node

    def check(self, symbols: dict):
        self.fieldType.check(symbols)

@dataclass
class MethodDecl(Node):
    name: str
    params: List["Param"] = field(default_factory=list)
    returnType: Optional[Node] = None
    body: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        func_symbols = {}
        func_symbols["return_type"] = self.returnType if self.returnType else None
        for param in self.params:
            param.check(func_symbols)
        for stmt in self.body:
            stmt.check(func_symbols)
        if self.returnType:
            self.returnType.check(func_symbols)

@dataclass
class DynvarSpec(Node):
    fields: List["DynvarField"] = field(default_factory=list)

    def check(self, symbols: dict):
        for field in self.fields:
            field.check(symbols)

@dataclass
class DynvarField(Node):
    name: str
    fieldType: Optional[Node] = None
    isManaged: bool = False

    def check(self, symbols: dict):
        if self.fieldType is not None:
            self.fieldType.check(symbols)

@dataclass
class Param(Node):
    name: str
    paramType: Node
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        name, paramType = attrs
        cname, _, ctype = coords
        return Param(name=name, paramType=paramType, pos=cname.start)

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Duplicate parameter '{self.name}'")
        symbols[self.name] = self.paramType
        self.paramType.check(symbols)

@dataclass
class FunctionDecl(Node):
    name: str
    params: List[Param] = field(default_factory=list)
    returnType: Optional[Node] = None
    locals: List[Node] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None  # позиция функции

    @pe.ExAction
    def create(attrs, coords, res_coord):
        func_name, params, retType, locals, body = attrs
        cfunc, _, _, _, cbrace_open, _, cbrace_close = coords  # упрощённо
        return FunctionDecl(name=func_name, params=params,
                            returnType=retType, locals=locals, body=body,
                            pos=cfunc.start)

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Function '{self.name}' already declared")
        func_symbols = {}
        func_symbols["return_type"] = self.returnType if self.returnType else None
        for param in self.params:
            param.check(func_symbols)
        for decl in self.locals:
            decl.check(func_symbols)
        for stmt in self.body:
            stmt.check(func_symbols)
        if self.returnType:
            self.returnType.check(func_symbols)

@dataclass
class GlobalVarDecl(Node):
    name: str
    varType: Node
    init: Optional[Node] = None
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        name, varType, init = attrs
        cname, _, ctype, csemicolon = coords
        return GlobalVarDecl(name=name, varType=varType, init=init, pos=cname.start)

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Global variable '{self.name}' already declared")
        self.varType.check(symbols)
        symbols[self.name] = self.varType
        if self.init:
            self.init.check(symbols)
            if self.init.type != self.varType:
                raise TypeMismatchError(self.init.pos, f"Type mismatch in initialization of '{self.name}'")

@dataclass
class LocalVarDecl(Node):
    name: str
    varType: Node
    init: Optional[Node] = None
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        name, varType = attrs  # если нет инициализации
        cname, _, ctype, csemicolon = coords
        return LocalVarDecl(name=name, varType=varType, init=None, pos=cname.start)

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Local variable '{self.name}' already declared")
        self.varType.check(symbols)
        symbols[self.name] = self.varType
        if self.init:
            self.init.check(symbols)
            if self.init.type != self.varType:
                raise TypeMismatchError(self.init.pos, f"Type mismatch in initialization of '{self.name}'")

@dataclass
class Assignment(Node):
    left: Node
    right: Node
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        left, right = attrs
        cleft, ceq, cright = coords
        return Assignment(left=left, right=right, pos=ceq.start)

    def check(self, symbols: dict):
        self.left.check(symbols)
        self.right.check(symbols)
        if self.left.type != self.right.type:
            raise TypeMismatchError(self.pos, f"Cannot assign {self.right.type} to {self.left.type}")

@dataclass
class IfStmt(Node):
    cond: Node
    thenBody: List[Node] = field(default_factory=list)
    elseIfs: List["ElseIf"] = field(default_factory=list)
    elseBody: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        cond, thenBody, elseIfs, elseBody = attrs
        cif, ccond, cthen_brace, cthenBody, celse_kw, celseBody = coords
        return IfStmt(cond=cond, thenBody=thenBody, elseIfs=elseIfs, elseBody=elseBody, pos=cif.start)

    def check(self, symbols: dict):
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(self.cond.pos, "Condition must be boolean")
        for stmt in self.thenBody:
            stmt.check(symbols)
        for else_if in self.elseIfs:
            else_if.check(symbols)
        for stmt in self.elseBody:
            stmt.check(symbols)

@dataclass
class ElseIf(Node):
    cond: Node
    body: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        cond, body = attrs
        ccond, _ = coords
        return ElseIf(cond=cond, body=body, pos=ccond.start)

    def check(self, symbols: dict):
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(self.cond.pos, "Condition must be boolean")
        for stmt in self.body:
            stmt.check(symbols)

@dataclass
class WhileStmt(Node):
    cond: Node
    body: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        cond, body = attrs
        cwhile, ccond, cdo, cbody = coords
        return WhileStmt(cond=cond, body=body, pos=cwhile.start)

    def check(self, symbols: dict):
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(self.cond.pos, "Condition must be boolean")
        for stmt in self.body:
            stmt.check(symbols)

@dataclass
class ReturnStmt(Node):
    value: Optional[Node] = None
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        if attrs:
            (value,) = attrs
            cret, cvalue = coords
            return ReturnStmt(value=value, pos=cret.start)
        else:
            cret, = coords
            return ReturnStmt(value=None, pos=cret.start)

    def check(self, symbols: dict):
        if "return_type" not in symbols:
            raise SemanticError(None, "Return outside function")
        expected = symbols["return_type"]
        if self.value:
            self.value.check(symbols)
            if self.value.type != expected:
                raise TypeMismatchError(self.value.pos, f"Expected {expected}, got {self.value.type}")
        elif expected is not None:
            raise SemanticError(self.pos, "Non-void function must return a value")

@dataclass
class FuncCall(Node):
    func: str
    args: List[Node] = field(default_factory=list)
    type: Optional[Node] = field(init=False)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        func_name, args = attrs
        cfunc, *cargs = coords
        return FuncCall(func=func_name, args=args, pos=cfunc.start)

    def check(self, symbols: dict):
        if self.func not in symbols:
            raise UndefinedError(self.pos, f"Undefined function '{self.func}'")
        func_decl = symbols[self.func]
        if not isinstance(func_decl, FunctionDecl):
            raise SemanticError(self.pos, f"'{self.func}' is not a function")
        for arg in self.args:
            arg.check(symbols)
        for arg, param in zip(self.args, func_decl.params):
            if arg.type != param.paramType:
                raise TypeMismatchError(arg.pos, f"Argument type mismatch in call to '{self.func}'")
        self.type = func_decl.returnType

@dataclass
class MethodCall(Node):
    obj: Node
    method: str
    args: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.obj.check(symbols)

@dataclass
class CloneCall(Node):
    src: Node
    dst: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.src.check(symbols)
        self.dst.check(symbols)

@dataclass
class AllocCall(Node):
    expr: Node
    extra: Optional[Node] = None
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.expr.check(symbols)
        if self.extra:
            self.extra.check(symbols)

@dataclass
class FinalCall(Node):
    obj: Node
    func: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.obj.check(symbols)
        self.func.check(symbols)

@dataclass
class NotExpr(Node):
    expr: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.expr.check(symbols)
        self.type = self.expr.type

@dataclass
class BoolType(Node):
    def check(self, symbols: dict):
        pass

@dataclass
class BoolConst(Node):
    value: bool
    type: BoolType = field(default_factory=BoolType)
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        pass

@dataclass
class Comparison(Node):
    op: str  # "<", ">", "==", и т.д.
    left: Node
    right: Node
    type: BoolType = field(default_factory=BoolType)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        left, op, right = attrs
        cleft, cop, cright = coords
        return Comparison(op=op, left=left, right=right, pos=cop.start)

    def check(self, symbols: dict):
        self.left.check(symbols)
        self.right.check(symbols)
        if self.left.type != self.right.type:
            raise TypeMismatchError(self.left.pos, f"Operands of '{self.op}' must have the same type")
        if not isinstance(self.left.type, (IntType, CharType)):
            raise InvalidOperationError(self.left.pos, f"Comparison '{self.op}' is not allowed for type {self.left.type}")
        self.type = BoolType()

@dataclass
class AExprAddress(Node):
    sub: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.sub.check(symbols)
        self.type = PointerType(self.sub.type)

@dataclass
class AExprNil(Node):
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.type = None

@dataclass
class AExprNum(Node):
    value: int
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        numTok, = attrs
        c, = coords
        return AExprNum(value=int(numTok), pos=c.start)

    def check(self, symbols: dict):
        self.type = IntType()

@dataclass
class AExprChar(Node):
    value: str
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        charTok, = attrs
        c, = coords
        return AExprChar(value=charTok, pos=c.start)

    def check(self, symbols: dict):
        self.type = CharType()

@dataclass
class AExprString(Node):
    value: str
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        sTok, = attrs
        c, = coords
        return AExprString(value=sTok, pos=c.start)

    def check(self, symbols: dict):
        self.type = CustomType("string")

@dataclass
class AExprVar(Node):
    name: str
    pos: Optional[pe.Position] = None
    type: Optional[object] = field(init=False, default=None)

    @pe.ExAction
    def create(attrs, coords, res_coord):
        name, = attrs
        c, = coords
        return AExprVar(name=name, pos=c.start)

    def check(self, symbols: dict):
        if self.name not in symbols:
            raise UndefinedError(self.pos, f"Undefined variable '{self.name}'")
        self.type = symbols[self.name]

@dataclass
class AExprPostfix(Node):
    atom: Node
    tail: List[Union[tuple, list]] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        atom, tail = attrs
        catom, *rest = coords
        return AExprPostfix(atom=atom, tail=tail, pos=catom.start)

    def check(self, symbols: dict):
        self.atom.check(symbols)
        current_type = self.atom.type
        for op in self.tail:
            if not (isinstance(op, (tuple, list)) and len(op) == 2):
                raise Exception("Неверный формат постфиксного оператора")
            op_name, operand = op
            if op_name == "index":
                operand.check(symbols)
                if not isinstance(current_type, ArrayType):
                    raise TypeMismatchError(self.atom.pos,
                        f"Индексирование допустимо только для массивов, получен тип {current_type}")
                current_type = current_type.element
            elif op_name == "field":
                if not isinstance(current_type, StructSpec):
                    raise TypeMismatchError(self.atom.pos,
                        f"Доступ к полю возможен только для структур, получен тип {current_type}")
                field_found = False
                for field in current_type.fields:
                    if field.name == operand:
                        current_type = field.fieldType
                        field_found = True
                        break
                if not field_found:
                    raise UndefinedError(self.atom.pos,
                        f"Поле '{operand}' не найдено в структуре {current_type}")
            else:
                raise Exception(f"Неизвестный постфиксный оператор: {op_name}")
        self.type = current_type

@dataclass
class AExprAddChain(Node):
    left: Node
    tail: List[tuple] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        left, tail = attrs
        cleft, *rest = coords
        return AExprAddChain(left=left, tail=tail, pos=cleft.start)

    def check(self, symbols: dict):
        self.left.check(symbols)
        current_type = self.left.type
        for op, operand in self.tail:
            operand.check(symbols)
            if not (isinstance(current_type, IntType) and isinstance(operand.type, IntType)):
                raise TypeMismatchError(self.left.pos,
                    f"Операция '{op}' требует тип int, а получены {current_type} и {operand.type}")
            current_type = IntType()
        self.type = current_type

@dataclass
class AExprMulChain(Node):
    left: Node
    tail: List[tuple] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        left, tail = attrs
        cleft, *rest = coords
        return AExprMulChain(left=left, tail=tail, pos=cleft.start)

    def check(self, symbols: dict):
        self.left.check(symbols)
        current_type = self.left.type
        for op, operand in self.tail:
            operand.check(symbols)
            if not (isinstance(current_type, IntType) and isinstance(operand.type, IntType)):
                raise TypeMismatchError(self.left.pos,
                    f"Операция '{op}' требует тип int, а получены {current_type} и {operand.type}")
            current_type = IntType()
        self.type = current_type

@dataclass
class BExprOrChain(Node):
    chain: List[Node] = field(default_factory=list)

@dataclass
class BExprAndChain(Node):
    chain: List[Node] = field(default_factory=list)

@dataclass
class IntType(Node):
    def check(self, symbols: dict):
        pass

@dataclass
class CharType(Node):
    def check(self, symbols: dict):
        pass

@dataclass
class PointerType(Node):
    base: Node

    def check(self, symbols: dict):
        self.base.check(symbols)

@dataclass
class ArrayType(Node):
    size: Optional[int]
    element: Node

    def check(self, symbols: dict):
        self.element.check(symbols)

@dataclass
class FuncType(Node):
    params: List[Node] = field(default_factory=list)
    returnType: Optional[Node] = None

@dataclass
class CustomType(Node):
    name: str

    def check(self, symbols: dict):
        pass

@dataclass
class ArrayInit(Node):
    values: List[Node] = field(default_factory=list)

@dataclass
class FieldAccess(Node):
    obj: Node
    fieldName: str
    type: Optional[object] = field(init=False)
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.obj.check(symbols)
        if not isinstance(self.obj.type, StructSpec):
            raise SemanticError(self.obj.pos,
                f"Expected struct type, got {self.obj.type}")
        struct_type = self.obj.type
        for field in struct_type.fields:
            if field.name == self.fieldName:
                self.type = field.fieldType
                return
        raise UndefinedError(self.pos,
            f"Struct '{struct_type}' has no field '{self.fieldName}'")

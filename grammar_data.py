import abc
import sys
sys.path.append('..')

import parser_edsl_new as pe
from lexer import *
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ============================================================================
# 1. AST – определения узлов синтаксического дерева
# ============================================================================

class SemanticError(Exception):
    def __init__(self, pos, message):
        self.pos = pos  # Позиция в исходном коде (Fragment)
        self.message = message

    def __str__(self):
        return f"Semantic error at {self.pos}: {self.message}"

class UndefinedError(SemanticError):
    pass

class TypeMismatchError(SemanticError):
    pass

class DuplicateDeclarationError(SemanticError):
    pass


class InvalidOperationError(SemanticError):
    pass

@dataclass
class Node(abc.ABC):
    def __init__(self):
        self.pos = None
        self.type = None

    @abc.abstractmethod
    def check(self, symbols: dict):
        """Проверка семантических правил для узла"""
        pass


@dataclass
class Program(Node):
    top_levels: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        """Проверка всей программы"""
        # Инициализируем таблицу символов для глобальной области видимости
        global_symbols = {}

        # Проверяем все топ-уровневые объявления
        for decl in self.top_levels:
            decl.check(global_symbols)

@dataclass
class TypeDecl(Node):
    name: str
    spec: Node

    def check(self, symbols):
        if self.name in symbols:
            raise DuplicateDeclarationError(None, f"Type '{self.name}' already declared")
        symbols[self.name] = self.spec
        self.spec.check(symbols)

@dataclass
class StructSpec(Node):
    fields: List["StructField"]

    def check(self, symbols):
        seen = set()
        for field in self.fields:
            if field.name in seen:
                raise DuplicateDeclarationError(None, f"Duplicate field '{field.name}'")
            seen.add(field.name)
            field.check(symbols)

@dataclass
class StructField(Node):
    name: str
    fieldType: Node

    def check(self, symbols):
        self.fieldType.check(symbols)

# Для описания класса (ООП)
@dataclass
class ClassSpec(Node):
    bases: List[str]
    members: List["ClassMember"]

@dataclass
class ClassMember(Node):
    name: str
    memberType: Optional[Node]  # если поле
    methodDecl: Optional[Node]  # если метод

# Узел для поля класса
@dataclass
class ClassField(Node):
    name: str
    fieldType: Node

@dataclass
class MethodDecl(Node):
    name: str
    params: List["Param"] = field(default_factory=list)
    returnType: Optional[Node] = None
    body: List[Node] = field(default_factory=list)

@dataclass
class ClassSpec(Node):
    bases: List[str] = field(default_factory=list)
    members: List[Node] = field(default_factory=list)

@dataclass
class DynvarSpec(Node):
    fields: List["DynvarField"]

    def check(self, symbols):
        for field in self.fields:
            field.check(symbols)

@dataclass
class DynvarField(Node):
    name: str
    fieldType: Optional[Node]
    isManaged: bool = False

    def check(self, symbols):
        if self.fieldType is not None:
            self.fieldType.check(symbols)

@dataclass
class Param(Node):
    name: str
    paramType: Node

    def check(self, symbols):
        if self.name in symbols:
            raise DuplicateDeclarationError(None, f"Duplicate parameter '{self.name}'")
        symbols[self.name] = self.paramType
        self.paramType.check(symbols)

@dataclass
class FunctionDecl(Node):
    name: str
    params: List[Param] = field(default_factory=list)
    returnType: Optional[Node] = None
    locals: List[Node] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)

    def check(self, symbols):
        if self.name in symbols:
            raise DuplicateDeclarationError(None, f"Function '{self.name}' already declared")
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
    init: Optional[Node]

    def check(self, symbols):
        if self.name in symbols:
            raise DuplicateDeclarationError(None, f"Global variable '{self.name}' already declared")

        self.varType.check(symbols)
        symbols[self.name] = self.varType

        if self.init:
            self.init.check(symbols)
            if self.init.type != self.varType:
                raise TypeMismatchError(None, f"Type mismatch in initialization of '{self.name}'")

@dataclass
class LocalVarDecl(Node):
    name: str
    varType: Node
    init: Optional[Node] = None

    def check(self, symbols):
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

    def check(self, symbols):
        self.left.check(symbols)
        self.right.check(symbols)

        if self.left.type != self.right.type:
            raise TypeMismatchError(None,
                                    f"Cannot assign {self.right.type} to {self.left.type}")


@dataclass
class IfStmt(Node):
    cond: Node
    thenBody: List[Node]
    elseIfs: List["ElseIf"]
    elseBody: List[Node]

    def check(self, symbols):
        # Проверка условия
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(
                self.cond.pos,
                "Condition must be boolean"
            )

        # Проверка then-блока
        for stmt in self.thenBody:
            stmt.check(symbols)

        # Проверка else-if веток
        for else_if in self.elseIfs:
            else_if.check(symbols)

        # Проверка else-блока
        for stmt in self.elseBody:
            stmt.check(symbols)

@dataclass
class ElseIf(Node):
    cond: Node
    body: List[Node]

@dataclass
class ElseIf(Node):
    cond: Node
    body: List[Node]

    def check(self, symbols):
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(
                self.cond.pos,
                "Condition must be boolean"
            )

        for stmt in self.body:
            stmt.check(symbols)

@dataclass
class WhileStmt(Node):
    cond: Node
    body: List[Node]

    def check(self, symbols):
        # Проверка условия
        self.cond.check(symbols)
        if not isinstance(self.cond.type, BoolType):
            raise TypeMismatchError(
                self.cond.pos,
                "Condition must be boolean"
            )

        # Проверка тела цикла
        for stmt in self.body:
            stmt.check(symbols)


@dataclass
class ReturnStmt(Node):
    value: Optional[Node]

    def check(self, symbols):
        if "return_type" not in symbols:
            raise SemanticError(None, "Return outside function")

        expected_type = symbols["return_type"]

        if self.value:
            self.value.check(symbols)
            if self.value.type != expected_type:
                raise TypeMismatchError(
                    self.value.pos,
                    f"Expected {expected_type}, got {self.value.type}"
                )
        elif expected_type is not None:
            raise SemanticError(None, "Non-void function must return a value")


@dataclass
class FuncCall(Node):
    func: str
    args: List[Node]
    type: Optional[Node] = field(init=False)

    def check(self, symbols):
        if self.func not in symbols:
            raise UndefinedError(None, f"Undefined function '{self.func}'")

        func = symbols[self.func]
        if not isinstance(func, FunctionDecl):
            raise SemanticError(None, f"'{self.func}' is not a function")

        # Проверка аргументов
        for arg in self.args:
            arg.check(symbols)

        # Проверка соответствия типов параметров
        for arg, param in zip(self.args, func.params):
            if arg.type != param.paramType:
                raise TypeMismatchError(None,
                                        f"Argument type mismatch in call to '{self.func}'")

        self.type = func.returnType

@dataclass
class MethodCall(Node):
    obj: Node
    method: str
    args: List[Node]

@dataclass
class CloneCall(Node):
    src: Node
    dst: Node

@dataclass
class AllocCall(Node):
    expr: Node
    extra: Optional[Node]

@dataclass
class FinalCall(Node):
    obj: Node
    func: Node

@dataclass
class NotExpr(Node):
    expr: Node

@dataclass
class BoolType(Node):
    def check(self, symbols):
        pass

@dataclass
class BoolConst(Node):
    value: bool
    type: BoolType = field(default_factory=BoolType)

    def check(self, symbols):
        pass  # Логическая константа всегда корректна


@dataclass
class Comparison(Node):
    op: str  # "<", ">", "==", и т.д.
    left: Node
    right: Node
    type: BoolType = field(default_factory=BoolType)

    def check(self, symbols):
        self.left.check(symbols)
        self.right.check(symbols)

        # Проверка совместимости типов операндов
        if self.left.type != self.right.type:
            raise TypeMismatchError(
                self.left.pos,
                f"Operands of '{self.op}' must have the same type"
            )

        # Проверка допустимых типов для сравнения
        if not isinstance(self.left.type, (IntType, CharType)):
            raise InvalidOperationError(
                self.left.pos,
                f"Comparison '{self.op}' is not allowed for type {self.left.type}"
            )

@dataclass
class AExprAddress(Node):
    sub: Node

@dataclass
class AExprNil(Node):
    pass

@dataclass
class AExprNum(Node):
    value: int

    def check(self, symbols):
        self.type = IntType()

@dataclass
class AExprChar(Node):
    value: str

@dataclass
class AExprString(Node):
    value: str

@dataclass
class AExprVar(Node):
    name: str
    type: Optional[Node] = field(init=False, default=None)

    def check(self, symbols):
        if self.name not in symbols:
            raise UndefinedError(None, f"Undefined variable '{self.name}'")
        self.type = symbols[self.name]

@dataclass
class AExprPostfix(Node):
    atom: Node
    tail: List[Union[tuple, list]]

    def check(self, symbols):
        self.atom.check(symbols)
        current_type = self.atom.type

        for op in self.tail:
            if not (isinstance(op, (tuple, list)) and len(op) == 2):
                raise Exception("Неверный формат постфиксного оператора")
            op_name, operand = op

            if op_name == "index":
                operand.check(symbols)
                if not isinstance(current_type, ArrayType):
                    raise TypeMismatchError(
                        self.atom.pos,
                        f"Индексирование допустимо только для массивов, получен тип {current_type}"
                    )
                current_type = current_type.element

            elif op_name == "field":
                if not isinstance(current_type, StructSpec):
                    raise TypeMismatchError(
                        self.atom.pos,
                        f"Доступ к полю возможен только для структур, получен тип {current_type}"
                    )
                # Ищем поле с именем operand
                field_found = False
                for field in current_type.fields:
                    if field.name == operand:
                        current_type = field.fieldType
                        field_found = True
                        break
                if not field_found:
                    raise UndefinedError(
                        self.atom.pos,
                        f"Поле '{operand}' не найдено в структуре {current_type}"
                    )
            else:
                raise Exception(f"Неизвестный постфиксный оператор: {op_name}")

        self.type = current_type

@dataclass
class AExprAddChain(Node):
    left: Node
    tail: List[tuple]

    def check(self, symbols):
        self.left.check(symbols)
        current_type = self.left.type

        for op, operand in self.tail:
            operand.check(symbols)
            if not (isinstance(current_type, IntType) and isinstance(operand.type, IntType)):
                raise TypeMismatchError(
                    self.left.pos,
                    f"Операция '{op}' требует тип int, а получены {current_type} и {operand.type}"
                )
            current_type = IntType()

        self.type = current_type

@dataclass
class AExprMulChain(Node):
    left: Node
    tail: List[tuple]

    def check(self, symbols):
        self.left.check(symbols)
        current_type = self.left.type

        for op, operand in self.tail:
            operand.check(symbols)
            if not (isinstance(current_type, IntType) and isinstance(operand.type, IntType)):
                raise TypeMismatchError(
                    self.left.pos,
                    f"Операция '{op}' требует тип int, а получены {current_type} и {operand.type}"
                )
            current_type = IntType()

        self.type = current_type

@dataclass
class BExprOrChain(Node):
    chain: List[Node]

@dataclass
class BExprAndChain(Node):
    chain: List[Node]

@dataclass
class IntType(Node):
    def check(self, symbols):
        pass

@dataclass
class CharType(Node):
    def check(self, symbols):
        pass

@dataclass
class PointerType(Node):
    base: Node

    def check(self, symbols):
        self.base.check(symbols)

@dataclass
class ArrayType(Node):
    size: Optional[int]  # Если None, то массив динамический
    element: Node

    def check(self, symbols):
        self.element.check(symbols)

@dataclass
class FuncType(Node):
    params: List[Node]
    returnType: Optional[Node]

@dataclass
class CustomType(Node):
    name: str

    def check(self, symbols):
        pass

# Для инициализации глобальных переменных (например, массивов)
@dataclass
class ArrayInit(Node):
    values: List[Node]

@dataclass
class FieldAccess(Node):
    obj: Node
    fieldName: str
    type: Optional[Node] = field(init=False)

    def check(self, symbols):
        self.obj.check(symbols)

        if not isinstance(self.obj.type, StructSpec):
            raise SemanticError(
                self.obj.pos,
                f"Expected struct type, got {self.obj.type}"
            )

        struct_type = self.obj.type
        for field in struct_type.fields:
            if field.name == self.fieldName:
                self.type = field.fieldType
                return

        raise UndefinedError(
            self.pos,
            f"Struct '{struct_type.type}' has no field '{self.fieldName}'"
        )
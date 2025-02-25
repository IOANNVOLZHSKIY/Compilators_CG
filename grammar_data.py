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

    def generate(self) -> str:
        raise NotImplementedError

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

    def generate(self) -> str:
        # Последовательность верхнеуровневых объявлений
        return "\n\n".join(decl.generate() for decl in self.top_levels)

@dataclass
class TypeDecl(Node):
    name: str
    spec: Node

    def check(self, symbols: dict):
        if self.name in symbols:
            raise DuplicateDeclarationError(self.pos, f"Type '{self.name}' already declared")
        symbols[self.name] = self.spec
        self.spec.check(symbols)

    def generate(self) -> str:
        return f";; type {self.name} defined as {self.spec.generate()}"

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

    def generate(self) -> str:
        fields_expr = "\n  ".join(field.generate() for field in self.fields)
        return f"(struct {fields_expr})"

@dataclass
class StructField(Node):
    name: str
    fieldType: Node

    def check(self, symbols: dict):
        self.fieldType.check(symbols)

    def generate(self) -> str:
        return f"({self.name} {self.fieldType.generate()})"

@dataclass
class ClassSpec(Node):
    bases: List[str] = field(default_factory=list)
    members: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        for member in self.members:
            member.check(symbols)

    def generate(self) -> str:
        bases_expr = " ".join(self.bases) if self.bases else ""
        members_expr = "\n  ".join(member.generate() for member in self.members)
        if bases_expr:
            return f"(class ({bases_expr})\n  {members_expr})"
        else:
            return f"(class\n  {members_expr})"

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

    def generate(self) -> str:
        if self.methodDecl:
            return self.methodDecl.generate()
        elif self.memberType:
            return f"(field {self.name} {self.memberType.generate()})"
        else:
            return f"(field {self.name} unknown)"

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

    def generate(self) -> str:
        params_expr = " ".join(p.generate() for p in self.params)
        body_expr = "\n  ".join(stmt.generate() for stmt in self.body)
        return f"(method {self.name} ({params_expr})\n  {body_expr}\n)"

@dataclass
class DynvarSpec(Node):
    fields: List["DynvarField"] = field(default_factory=list)

    def check(self, symbols: dict):
        for field in self.fields:
            field.check(symbols)

    def generate(self) -> str:
        fields_expr = "\n  ".join(field.generate() for field in self.fields)
        return f"(dynvar\n  {fields_expr}\n)"

@dataclass
class DynvarField(Node):
    name: str
    fieldType: Optional[Node] = None
    isManaged: bool = False

    def check(self, symbols: dict):
        if self.fieldType is not None:
            self.fieldType.check(symbols)

    def generate(self) -> str:
        if self.fieldType:
            return f"({self.name} {self.fieldType.generate()})"
        else:
            return f"({self.name} {'managed' if self.isManaged else 'default'})"

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

    def generate(self) -> str:
        return self.name

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

    def generate(self) -> str:
        params_expr = " ".join(p.generate() for p in self.params)
        locals_expr = " ".join(loc.generate() for loc in self.locals) if self.locals else ""
        body_expr = "\n  ".join(stmt.generate() for stmt in self.body)
        var_part = f"(var {locals_expr})" if locals_expr else ""
        return f"(function {self.name} ({params_expr})\n  {var_part}\n  {body_expr}\n)"

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

    def generate(self) -> str:
        if self.init:
            init_expr = self.init.generate()
            return f"(var {self.name} {init_expr})"
        else:
            return f"(var {self.name} 0)"

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

    def generate(self) -> str:
        if self.init:
            init_expr = self.init.generate()
            return f"({self.name} {init_expr})"
        else:
            return f"({self.name} 0)"

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

    def generate(self) -> str:
        left_expr = self.left.generate()
        right_expr = self.right.generate()
        return f"({left_expr} \"=\" {right_expr})"

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

    def generate(self) -> str:
        cond_expr = self.cond.generate()
        then_expr = "\n  ".join(stmt.generate() for stmt in self.thenBody)
        result = f"(if {cond_expr}\n  ({then_expr})"
        if self.elseIfs:
            elif_exprs = "\n  ".join(e.generate() for e in self.elseIfs)
            result += f"\n  {elif_exprs}"
        if self.elseBody:
            else_expr = "\n  ".join(stmt.generate() for stmt in self.elseBody)
            result += f"\n else ({else_expr})"
        result += ")"
        return result

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

    def generate(self) -> str:
        cond_expr = self.cond.generate()
        body_expr = "\n  ".join(stmt.generate() for stmt in self.body)
        return f"(elseif {cond_expr} ({body_expr}))"

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

    def generate(self) -> str:
        cond_expr = self.cond.generate()
        body_expr = "\n  ".join(stmt.generate() for stmt in self.body)
        return f"(while ({cond_expr})\n  ({body_expr}))"

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

    def generate(self) -> str:
        if self.value:
            return f"(return {self.value.generate()})"
        else:
            return "(return)"

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

    def generate(self) -> str:
        args_expr = " ".join(arg.generate() for arg in self.args)
        return f"(call {self.func} {args_expr})"

@dataclass
class MethodCall(Node):
    obj: Node
    method: str
    args: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.obj.check(symbols)
        for arg in self.args:
            arg.check(symbols)

    def generate(self) -> str:
        args_expr = " ".join(arg.generate() for arg in self.args)
        return f"(method-call {self.obj.generate()} {self.method} {args_expr})"

@dataclass
class CloneCall(Node):
    src: Node
    dst: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.src.check(symbols)
        self.dst.check(symbols)

    def generate(self) -> str:
        return f"(clone {self.src.generate()} {self.dst.generate()})"

@dataclass
class AllocCall(Node):
    expr: Node
    extra: Optional[Node] = None
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.expr.check(symbols)
        if self.extra:
            self.extra.check(symbols)

    def generate(self) -> str:
        if self.extra:
            return f"(alloc {self.expr.generate()} {self.extra.generate()})"
        else:
            return f"(alloc {self.expr.generate()})"

@dataclass
class FinalCall(Node):
    obj: Node
    func: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.obj.check(symbols)
        self.func.check(symbols)

    def generate(self) -> str:
        return f"(final {self.obj.generate()} {self.func.generate()})"

@dataclass
class NotExpr(Node):
    expr: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.expr.check(symbols)

    def generate(self) -> str:
        return f"(not {self.expr.generate()})"

@dataclass
class BoolType(Node):
    def check(self, symbols: dict):
        pass

    def generate(self) -> str:
        return "bool"

@dataclass
class BoolConst(Node):
    value: bool
    type: BoolType = field(default_factory=BoolType)
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        pass

    def generate(self) -> str:
        return "true" if self.value else "false"

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
            print(self.left.type)
            print(self.right.type)
            raise TypeMismatchError(self.left.pos, f"Operands of '{self.op}' must have the same type")
        if not isinstance(self.left.type, (IntType, CharType)):
            raise InvalidOperationError(self.left.pos, f"Comparison '{self.op}' is not allowed for type {self.left.type}")
        self.type = BoolType()

    def generate(self) -> str:
        return f"({self.left.generate()} {self.op} {self.right.generate()})"

@dataclass
class AExprAddress(Node):
    sub: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.sub.check(symbols)

    def generate(self) -> str:
        return f"(L {self.sub.generate()})"

@dataclass
class AExprNil(Node):
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.type = None

    def generate(self) -> str:
        return "nil"

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

    def generate(self) -> str:
        return str(self.value)

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

    def generate(self) -> str:
        return f"'{self.value}'"

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

    def generate(self) -> str:
        return f"\"{self.value}\""

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

    def generate(self) -> str:
        return self.name

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

    def generate(self) -> str:
        result = self.atom.generate()
        for op in self.tail:
            if isinstance(op, (tuple, list)) and len(op) == 2:
                op_name, operand = op
                if op_name == "index":
                    result = f"({result} [ {operand.generate()} ])"
                elif op_name == "field":
                    result = f"({result} . {operand})"
        return result

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

    def generate(self) -> str:
        result = self.left.generate()
        for op, operand in self.tail:
            result = f"({result} {op} {operand.generate()})"
        return result

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

    def generate(self) -> str:
        result = self.left.generate()
        for op, operand in self.tail:
            result = f"({result} {op} {operand.generate()})"
        return result

@dataclass
class BExprOrChain(Node):
    chain: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        for expr in self.chain:
            expr.check(symbols)

    def generate(self) -> str:
        if not self.chain:
            return "false"
        exprs = " ".join(expr.generate() for expr in self.chain)
        return f"(or {exprs})"

@dataclass
class BExprAndChain(Node):
    chain: List[Node] = field(default_factory=list)

    def check(self, symbols: dict):
        for expr in self.chain:
            expr.check(symbols)

    def generate(self) -> str:
        if not self.chain:
            return "true"
        exprs = " ".join(expr.generate() for expr in self.chain)
        return f"(and {exprs})"

@dataclass
class IntType(Node):
    def check(self, symbols: dict):
        pass

    def generate(self) -> str:
        return "int"

@dataclass
class CharType(Node):
    def check(self, symbols: dict):
        pass

    def generate(self) -> str:
        return "char"

@dataclass
class PointerType(Node):
    base: Node

    def check(self, symbols: dict):
        self.base.check(symbols)

    def generate(self) -> str:
        return f"(* {self.base.generate()})"

@dataclass
class ArrayType(Node):
    size: Optional[int]
    element: Node

    def check(self, symbols: dict):
        self.element.check(symbols)

    def generate(self) -> str:
        if self.size is not None:
            return f"([{self.size}] {self.element.generate()})"
        else:
            return f"([] {self.element.generate()})"

@dataclass
class FuncType(Node):
    params: List[Node] = field(default_factory=list)
    returnType: Optional[Node] = None

    def check(self, symbols: dict):
        for param in self.params:
            param.check(symbols)
        if self.returnType:
            self.returnType.check(symbols)

    def generate(self) -> str:
        params_expr = " ".join(p.generate() for p in self.params)
        ret_expr = self.returnType.generate() if self.returnType else ""
        return f"(func-type ({params_expr}) {ret_expr})"

@dataclass
class CustomType(Node):
    name: str

    def check(self, symbols: dict):
        pass

    def generate(self) -> str:
        return self.name

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
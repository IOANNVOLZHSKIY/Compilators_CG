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

    def is_lvalue(self) -> bool:
        return False

    @staticmethod
    def generate_rval(node: "Node") -> str:
        if isinstance(node, AExprAddress):
            return node.generate()
        if node.is_lvalue():
            return AExprAddress(sub=node, pos=getattr(node, 'pos', None)).generate()
        return node.generate()


class SemanticError(pe.Error):
    def __init__(self, pos, message):
        self.atom.name = pos
        self.__message = message

    @property
    def message(self):
        return self.__message

    def __str__(self):
        return f"Semantic error at {self.atom.name}: {self.__message}"


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
            try:
                decl.check(symbols)
            except Exception as e:
                print(e)

    def generate(self) -> str:
        return "\n\n".join(decl.generate() for decl in self.top_levels)

@dataclass
class TypeDecl(Node):
    name: str
    spec: Node

    def check(self, symbols: dict):
        if self.name in symbols:
            print(f"Semantic error at {getattr(self, 'pos', None)}: Type '{self.name}' already declared")
        else:
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
                print(f"Semantic error at {field.atom.name}: Duplicate field '{field.name}'")
            else:
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
            print(f"Semantic error at {self.atom.name}: Duplicate parameter '{self.name}'")
        else:
            symbols[self.name] = self.paramType
        self.paramType.check(symbols)

    def generate(self) -> str:
        return self.name


@dataclass
class FunctionDecl(Node):
    name: str
    params: List["Param"] = field(default_factory=list)
    returnType: Optional[Node] = None
    locals: List[Node] = field(default_factory=list)
    body: List[Node] = field(default_factory=list)
    pos: Optional[pe.Position] = None

    @pe.ExAction
    def create(attrs, coords, res_coord):
        func_name, params, retType, locals, body = attrs
        cfunc, _, _, _, cbrace_open, _, cbrace_close = coords
        return FunctionDecl(name=func_name, params=params,
                            returnType=retType, locals=locals, body=body,
                            pos=cfunc.start)

    def check(self, symbols: dict):
        if self.name in symbols:
            print(f"Semantic error at {self.atom.name}: Function '{self.name}' already declared")
        else:
            symbols[self.name] = self
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
            print(f"Semantic error at {self.atom.name}: Global variable '{self.name}' already declared")
        else:
            self.varType.check(symbols)
            symbols[self.name] = self.varType
        if self.init:
            self.init.check(symbols)
            if hasattr(self.init, 'type') and self.init.type != self.varType:
                print(f"Semantic error at {self.init.atom.name}: Type mismatch in initialization of '{self.name}'")

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
            print(f"Semantic error at {self.atom.name}: Local variable '{self.name}' already declared")
        else:
            self.varType.check(symbols)
            symbols[self.name] = self.varType
        if self.init:
            self.init.check(symbols)
            if hasattr(self.init, 'type') and self.init.type != self.varType:
                print(f"Semantic error at {self.init.atom.name}: Type mismatch in initialization of '{self.name}'")

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
        # Проверяем, что левая часть является левым значением
        if not self.left.is_lvalue():
            print(f"Semantic error at {self.atom.name}: Left-hand side of assignment must be an lvalue")
        # Если правая часть – левое значение, необходимо разыменовать её
        if self.right.is_lvalue():
            # Автоматически оборачиваем правую часть в операцию разыменования
            self.right = AExprAddress(sub=self.right, pos=self.right.atom.name)
            self.right.check(symbols)
        # Ограничение: присваиваемое значение должно быть "одним словом"
        if isinstance(getattr(self.right, 'type', None), (ArrayType, StructSpec)):
            print(f"Semantic error at {self.right.atom.name}: Assigned value must be a single word")
        # Проверяем типовую совместимость
        if hasattr(self.left, 'type') and hasattr(self.right, 'type') and self.left.type != self.right.type:
            print(f"Semantic error at {self.atom.name}: Cannot assign {self.right.type} to {self.left.type}")

    def generate(self) -> str:
        left_expr = self.left.generate()
        right_expr = Node.generate_rval(self.right)
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
        if not isinstance(getattr(self.cond, 'type', None), BoolType):
            print(f"Semantic error at {self.cond.atom.name}: Condition must be boolean")
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
        if not isinstance(getattr(self.cond, 'type', None), BoolType):
            print(f"Semantic error at {self.cond.atom.name}: Condition must be boolean")
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
        if not isinstance(getattr(self.cond, 'type', None), BoolType):
            print(f"Semantic error at {self.cond.atom.name}: Condition must be boolean")
        for stmt in self.body:
            stmt.check(symbols)

    def generate(self) -> str:
        cond_expr = Node.generate_rval(self.cond)
        body_expr = "\n  ".join(stmt.generate() for stmt in self.body)
        return f"(while {cond_expr}\n  {body_expr})"


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
            print("Semantic error: Return outside function")
            return
        expected = symbols["return_type"]
        if self.value:
            self.value.check(symbols)
            if hasattr(self.value, 'type') and self.value.type != expected:
                print(f"Semantic error at {self.value.atom.name}: Expected {expected}, got {self.value.type}")
        elif expected is not None:
            print(f"Semantic error at {self.atom.name}: Non-void function must return a value")

    def generate(self) -> str:
        if self.value:
            return f"(return {Node.generate_rval(self.value)})"
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
            print(f"Semantic error at {self.atom.name}: Undefined function '{self.func}'")
            return
        func_decl = symbols[self.func]
        if not isinstance(func_decl, FunctionDecl):
            print(f"Semantic error at {self.atom.name}: '{self.func}' is not a function")
            return
        for arg in self.args:
            arg.check(symbols)
            # Разрешённые типы аргументов: int, указатели и пользовательские (custom) типы
            if not (isinstance(arg.type, IntType) or isinstance(arg.type, PointerType) or isinstance(arg.type, CustomType)):
                print(f"Semantic error at {arg.atom.name}: Invalid argument type {arg.type} in function call '{self.func}'")
        for arg, param in zip(self.args, func_decl.params):
            if hasattr(arg, 'type') and arg.type != param.paramType:
                print(f"Semantic error at {arg.atom.name}: Argument type mismatch in call to '{self.func}'")
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
        if hasattr(self.left, 'type') and hasattr(self.right, 'type') and self.left.type != self.right.type:
            print(f"Semantic error at {self.left.atom.name}: Operands of '{self.op}' must have the same type")
        if not isinstance(getattr(self.left, 'type', None), (IntType, CharType)):
            print(f"Semantic error at {self.left.atom.name}: Comparison '{self.op}' is not allowed for type {self.left.type}")
        self.type = BoolType()

    def generate(self) -> str:
        return f"({Node.generate_rval(self.left)} {self.op} {Node.generate_rval(self.right)})"


@dataclass
class AExprAddress(Node):
    sub: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.sub.check(symbols)
    def generate(self) -> str:
        return f"(L {self.sub.generate()})"
    def is_lvalue(self) -> bool:
        # Результат операции разыменования (L ...) является левым значением
        return True


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
            print(f"Semantic error at {self.atom.name}: Undefined variable '{self.name}'")
        else:
            self.type = symbols[self.name]
    def generate(self) -> str:
        return self.name
    def is_lvalue(self) -> bool:
        return True


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
                print("Semantic error: Invalid postfix operator format")
                continue
            op_name, operand = op
            if op_name == "index":
                operand.check(symbols)
                if not isinstance(current_type, ArrayType):
                    print(f"Semantic error at {self.atom.atom.name}: Indexing is allowed only for arrays, got {current_type}")
                else:
                    current_type = current_type.element
            elif op_name == "field":
                if not isinstance(current_type, StructSpec):
                    print(f"Semantic error at {self.atom.atom.name}: Field access allowed only for structs, got {current_type}")
                else:
                    field_found = False
                    for field in current_type.fields:
                        if field.name == operand:
                            current_type = field.fieldType
                            field_found = True
                            break
                    if not field_found:
                        print(f"Semantic error at {self.atom.atom.name}: Field '{operand}' not found in {current_type}")
            else:
                print(f"Semantic error: Unknown postfix operator: {op_name}")
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

    def is_lvalue(self) -> bool:
        # Если базовое выражение является левым значением и все постфиксные операции – индекс или поле, то результат – левое значение
        if not self.atom.is_lvalue():
            return False
        for op in self.tail:
            if not (isinstance(op, (tuple, list)) and len(op) == 2 and op[0] in ("index", "field")):
                return False
        return True


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
                print(f"Semantic error at {self.left.atom.name}: Operation '{op}' requires int, got {current_type} and {operand.type}")
            current_type = IntType()
        self.type = current_type

    def generate(self) -> str:
        result = Node.generate_rval(self.left)
        for op, operand in self.tail:
            result = f"({result} {op} {Node.generate_rval(operand)})"
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
                print(f"Semantic error at {self.left.atom.name}: Operation '{op}' requires int, got {current_type} and {operand.type}")
            current_type = IntType()
        self.type = current_type

    def generate(self) -> str:
        result = Node.generate_rval(self.left)
        for op, operand in self.tail:
            result = f"({result} {op} {Node.generate_rval(operand)})"
        return result

@dataclass
class AExprAddressOf(Node):
    sub: Node
    pos: Optional[pe.Position] = None

    def check(self, symbols: dict):
        self.sub.check(symbols)
        if not self.sub.is_lvalue():
            print(f"Semantic error at {self.atom.name}: Cannot take address of a non-lvalue")
        # Типом будет указатель на тип операнда
        self.type = PointerType(self.sub.type)

    def generate(self) -> str:
        return f"(& {self.sub.generate()})"

    def is_lvalue(self) -> bool:
        # Результат взятия адреса не является левым значением
        return False

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
            print(f"Semantic error at {self.obj.atom.name}: Expected struct type, got {self.obj.type}")
            return
        struct_type = self.obj.type
        for field in struct_type.fields:
            if field.name == self.fieldName:
                self.type = field.fieldType
                return
        print(f"Semantic error at {self.atom.name}: Struct '{struct_type}' has no field '{self.fieldName}'")
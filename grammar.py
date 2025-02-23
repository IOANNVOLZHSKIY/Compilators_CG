from grammar_data import *

ProgramNT = pe.NonTerminal("ProgramNT")
ProgramStart = pe.NonTerminal("ProgramStart")
TopLevel = pe.NonTerminal("TopLevel")
TypeDeclNT = pe.NonTerminal("TypeDeclNT")
FunctionDeclNT = pe.NonTerminal("FunctionDeclNT")
TypeSpecNT = pe.NonTerminal("TypeSpecNT")
StructSpecNT = pe.NonTerminal("StructSpecNT")
ClassSpecNT = pe.NonTerminal("ClassSpecNT")
TypeNT = pe.NonTerminal("TypeNT")
StructFieldsNT = pe.NonTerminal("StructFieldsNT")
StructFieldNT = pe.NonTerminal("StructFieldNT")
GlobalVarDeclNT = pe.NonTerminal("GlobalVarDeclNT")
OptParamListNT = pe.NonTerminal("OptParamListNT")
OptTypeNT = pe.NonTerminal("OptTypeNT")
TypeOpt = pe.NonTerminal("TypeOpt")
OptInitGlobalVarNT = pe.NonTerminal("OptInitGlobalVarNT")
InitGlobalVarNT = pe.NonTerminal("InitGlobalVarNT")
LocalVarDeclsNT = pe.NonTerminal("LocalVarDeclsNT")
StatementsNT = pe.NonTerminal("StatementsNT")
ValueListNT = pe.NonTerminal("ValueListNT")
AExprNT = pe.NonTerminal("AExprNT")
ValueListTailNT = pe.NonTerminal("ValueListTailNT")
ParamListNT = pe.NonTerminal("ParamListNT")
ParamNT = pe.NonTerminal("ParamNT")
ParamListTailNT = pe.NonTerminal("ParamListTailNT")
LocalVarDeclNT = pe.NonTerminal("LocalVarDeclNT")
StatementNT = pe.NonTerminal("StatementNT")
AssignmentNT = pe.NonTerminal("AssignmentNT")
IfStmtNT = pe.NonTerminal("IfStmtNT")
BExprNT = pe.NonTerminal("BExprNT")
ElseIfChainNT = pe.NonTerminal("ElseIfChainNT")
OptElseNT = pe.NonTerminal("OptElseNT")
WhileStmtNT = pe.NonTerminal("WhileStmtNT")
ReturnStmtNT = pe.NonTerminal("ReturnStmtNT")
OptAExprNT = pe.NonTerminal("OptAExprNT")
FuncCallStmtNT = pe.NonTerminal("FuncCallStmtNT")
OptCommaAExprNT = pe.NonTerminal("OptCommaAExprNT")
MethodCallStmtNT = pe.NonTerminal("MethodCallStmtNT")
OptArgListNT = pe.NonTerminal("OptArgListNT")
CloneCallNT = pe.NonTerminal("CloneCallNT")
AllocCallNT = pe.NonTerminal("AllocCallNT")
FinalCallNT = pe.NonTerminal("FinalCallNT")
ArgListNT = pe.NonTerminal("ArgListNT")
ArgListTailNT = pe.NonTerminal("ArgListTailNT")
PrimaryAExprNT = pe.NonTerminal("PrimaryAExprNT")
PostfixTailItemNT = pe.NonTerminal("PostfixTailItemNT")
PostfixTailNT = pe.NonTerminal("PostfixTailNT")
AExprPostfixNT = pe.NonTerminal("AExprPostfixNT")
BaseClassListNT = pe.NonTerminal("BaseClassListNT")
BaseClassTailNT = pe.NonTerminal("BaseClassTailNT")
OptionalBaseClassNT = pe.NonTerminal("OptionalBaseClassNT")
ClassFieldOrMethodNT = pe.NonTerminal("ClassFieldOrMethodNT")
ClassFieldOrMethodListNT = pe.NonTerminal("ClassFieldOrMethodListNT")
MethodDeclNT = pe.NonTerminal("MethodDeclNT")
ClassMemberNT = pe.NonTerminal("ClassMemberNT")
ClassMemberListNT = pe.NonTerminal("ClassMemberListNT")
OptionalBaseListNT = pe.NonTerminal("OptionalBaseListNT")
BaseClassListTailNT = pe.NonTerminal("BaseClassListTailNT")
AExprMulChainNT = pe.NonTerminal("AExprMulChain")
AExprAddChainNT = pe.NonTerminal("AExprAddChain")
DynvarFieldNT = pe.NonTerminal("DynvarFieldNT")
DynvarFieldListNT = pe.NonTerminal("DynvarFieldListNT")
DynvarSpecNT = pe.NonTerminal("DynvarSpecNT")
AssignExprNT = pe.NonTerminal("AssignExprNT")

# Определяем ε-производство
Eps = pe.NonTerminal("Eps")
Eps |= (lambda: None)

# --- Программа ---
ProgramNT |= (TopLevel, ProgramNT, lambda hd, tl: [hd] + tl)
ProgramNT |= (Eps, lambda: [])

ProgramStart |= (ProgramNT, lambda tops: Program(top_levels=tops))

# --- TopLevel --- (топ-уровневые объявления: функция, тип, глоб. переменная)
TopLevel |= (FunctionDeclNT, lambda fd: fd)
TopLevel |= (TypeDeclNT, lambda td: td)
TopLevel |= (GlobalVarDeclNT, lambda gv: gv)
#pfield, const

# --- Объявление типа ---
TypeDeclNT |= (T.TYPE, T.IDENT, TypeSpecNT,
               lambda _type, idTok, spec: TypeDecl(name=idTok, spec=spec))

# --- Классы ---
# Правило для поля:
ClassMemberNT |= (T.IDENT, TypeNT,
                  lambda idTok, typ: ClassField(name=idTok, fieldType=typ))
# Правило для метода:
MethodDeclNT |= (T.FUNC, T.IDENT, T.LP, OptParamListNT, T.RP, OptTypeNT, T.LBRACE, StatementsNT, T.RBRACE,
                 lambda _func, idTok, params, retT, stmts:
                     MethodDecl(name=idTok, params=params, returnType=retT, body=stmts))
ClassMemberNT |= (MethodDeclNT, lambda m: m)

# Нетерминал для списка членов класса:
ClassMemberListNT |= (ClassMemberNT, ClassMemberListNT,
                      lambda head, tail: [head] + tail)
ClassMemberListNT |= (Eps, lambda: [])

# Правило для спецификации класса (без базовых классов):
ClassSpecNT |= (T.CLASS, Eps, T.LBRACE, ClassMemberListNT, T.RBRACE,
                lambda _class, eps, members: ClassSpec(bases=[], members=members))

# Нетерминалы для базовых классов
OptionalBaseListNT |= (T.LP, BaseClassListNT, T.RP, lambda bases: bases)
OptionalBaseListNT |= (Eps, lambda: [])

BaseClassListNT |= (T.IDENT, BaseClassListTailNT, lambda idTok, tail: [idTok] + tail)
BaseClassListNT |= (T.IDENT, lambda idTok: [idTok])

BaseClassListTailNT |= (T.COMMA, T.IDENT, BaseClassListTailNT,
                         lambda idTok, tail: [idTok] + tail)
BaseClassListTailNT |= (Eps, lambda: [])

# Правило для спецификации класса с базовыми классами:
ClassSpecNT |= (T.CLASS, OptionalBaseListNT, T.LBRACE, ClassMemberListNT, T.RBRACE,
                lambda _class, bases, members: ClassSpec(bases=bases, members=members))

# Правило для динамического типа:
DynvarFieldNT |= (T.IDENT, TypeNT,
                  lambda idTok, typ: DynvarField(name=idTok, fieldType=typ, isManaged=False))
DynvarFieldNT |= (T.IDENT, lambda idTok: DynvarField(name=idTok, fieldType=None, isManaged=True))

DynvarFieldListNT |= (DynvarFieldNT, DynvarFieldListNT,
                      lambda head, tail: [head] + tail)
DynvarFieldListNT |= (Eps, lambda: [])

DynvarSpecNT |= (T.DYNVAR, T.LBRACE, DynvarFieldListNT, T.RBRACE,
                 lambda _dynvar, fields: DynvarSpec(fields=fields))

# --- Спецификация типа ---
TypeSpecNT |= (StructSpecNT, lambda ss: ss)
TypeSpecNT |= (TypeNT, lambda t: t)
TypeSpecNT |= (ClassSpecNT, lambda cs: cs)
TypeSpecNT |= (DynvarSpecNT, lambda ds: ds)

# --- StructSpec ---
StructSpecNT |= (T.LBRACE, StructFieldsNT, T.RBRACE,
                lambda fields: StructSpec(fields=fields))

StructFieldsNT |= (StructFieldNT, StructFieldsNT, lambda field, fields: [field] + fields)
StructFieldsNT |= (Eps, lambda: [])

StructFieldNT |= (T.IDENT, TypeNT,
                  lambda idTok, typ: StructField(name=idTok, fieldType=typ))

# --- TypeNT --- (определение типов)
TypeNT |= (T.INT, lambda tok: IntType())
TypeNT |= (T.CHAR_KW, lambda tok: CharType())
TypeNT |= (T.MUL, TypeNT, lambda typ: PointerType(base=typ))
TypeNT |= (T.LSQUARE, T.NUMBER, T.RSQUARE, TypeNT,
           lambda numTok, typ: ArrayType(size=int(numTok), element=typ))
TypeNT |= (T.SYM, TypeNT, lambda typ: ArrayType(size=None, element=typ))
TypeNT |= (T.FUNC, T.LP, OptParamListNT, T.RP, TypeOpt,
           lambda func, params, retT: FuncType(params=params, returnType=retT))
TypeNT |= (T.IDENT, lambda idTok: CustomType(name=idTok))

# --- TypeOpt --- (необязательный тип, например, возвращаемый тип функции)
TypeOpt |= (TypeNT, lambda t: t)
TypeOpt |= (Eps, lambda: None)

# --- Глобальные переменные ---
GlobalVarDeclNT |= (T.VAR, T.IDENT, TypeNT, OptInitGlobalVarNT,
                    lambda _var, idTok, typ, init: GlobalVarDecl(name=idTok, varType=typ, init=init))

OptInitGlobalVarNT |= (T.EQ, InitGlobalVarNT, lambda iv: iv)
OptInitGlobalVarNT |= (Eps, lambda: None)

InitGlobalVarNT |= (T.LBRACE, ValueListNT, T.RBRACE,
                   lambda vals: ArrayInit(values=vals))
InitGlobalVarNT |= (AExprNT, lambda expr: expr)

ValueListNT |= (AExprNT, ValueListTailNT, lambda first, tail: [first] + tail)
ValueListNT |= (AExprNT, lambda expr: [expr])

ValueListTailNT |= (T.COMMA, AExprNT, ValueListTailNT, lambda expr, tail: [expr] + tail)
ValueListTailNT |= (Eps, lambda: [])

# --- Функция ---
FunctionDeclNT |= (T.FUNC, T.IDENT, T.LP, OptParamListNT, T.RP, OptTypeNT, T.LBRACE, LocalVarDeclsNT, StatementsNT, T.RBRACE,
                   lambda idTok, params, retT, locals, stats:
                        FunctionDecl(name=idTok, params=params, returnType=retT, locals=locals, body=stats))

OptParamListNT |= (ParamListNT, lambda lst: lst)
OptParamListNT |= (Eps, lambda: [])

ParamListNT |= (ParamNT, ParamListTailNT, lambda p, tail: [p] + tail)
ParamListNT |= (ParamNT, lambda p: [p])

ParamListTailNT |= (T.COMMA, ParamNT, ParamListTailNT, lambda p, tail: [p] + tail)
ParamListTailNT |= (Eps, lambda: [])

ParamNT |= (T.IDENT, TypeNT, lambda idTok, typ: Param(name=idTok, paramType=typ))

OptTypeNT |= (TypeNT, lambda t: t)
OptTypeNT |= (Eps, lambda: "")

LocalVarDeclsNT |= (LocalVarDeclNT, LocalVarDeclsNT, lambda decl, decls: [decl] + decls)
LocalVarDeclsNT |= (Eps, lambda: [])

LocalVarDeclNT |= (T.VAR, T.IDENT, TypeNT,
                    lambda _var, idTok, typ: LocalVarDecl(name=idTok, varType=typ))

# --- Операторы (Statements) ---
StatementsNT |= (StatementNT, StatementsNT, lambda st, tail: [st] + tail)
StatementsNT |= (Eps, lambda: [])

StatementNT |= (AssignmentNT, lambda a: a)
StatementNT |= (IfStmtNT, lambda i: i)
StatementNT |= (WhileStmtNT, lambda w: w)
StatementNT |= (ReturnStmtNT, lambda r: r)
StatementNT |= (FuncCallStmtNT, lambda fc: fc)
StatementNT |= (MethodCallStmtNT, lambda mc: mc)
StatementNT |= (CloneCallNT, lambda cc: cc)
StatementNT |= (AllocCallNT, lambda ac: ac)
StatementNT |= (FinalCallNT, lambda fc: fc)

# Assignment: AExprNT EQ AExprNT
AssignmentNT |= (AExprNT, T.EQ, AExprNT,
                 lambda left, right: Assignment(left=left, right=right))

# IfStmt: IF BExprNT LBRACE StatementsNT RBRACE ElseIfChainNT OptElseNT
IfStmtNT |= (T.IF, BExprNT, T.LBRACE, StatementsNT, T.RBRACE, ElseIfChainNT, OptElseNT,
             lambda _if, cond, thenStmts, elifs, elseStmts:
                 IfStmt(cond=cond, thenBody=thenStmts, elseIfs=elifs, elseBody=elseStmts))

ElseIfChainNT |= (T.ELSE, T.IF, BExprNT, T.LBRACE, StatementsNT, T.RBRACE, ElseIfChainNT,
                  lambda _else, _if, cond, stmts, tail: [ElseIf(cond=cond, body=stmts)] + tail)
ElseIfChainNT |= (Eps, lambda: [])

OptElseNT |= (T.ELSE, T.LBRACE, StatementsNT, T.RBRACE,
              lambda _else, stmts: stmts)
OptElseNT |= (Eps, lambda: [])

# WhileStmt: WHILE BExprNT LBRACE StatementsNT RBRACE
WhileStmtNT |= (T.WHILE, BExprNT, T.LBRACE, StatementsNT, T.RBRACE,
                lambda _while, cond, stmts: WhileStmt(cond=cond, body=stmts))

# ReturnStmt: RETURN OptAExprNT
ReturnStmtNT |= (T.RETURN, OptAExprNT, lambda _ret, expr: ReturnStmt(value=expr))

OptAExprNT |= (AExprNT, lambda expr: expr)
#OptAExprNT |= (Eps, lambda: "")

# FuncCallStmt: IDENT LP OptArgListNT RP
FuncCallStmtNT |= (T.IDENT, T.LP, OptArgListNT, T.RP,
                   lambda idTok, args: FuncCall(func=idTok, args=args))

# MethodCallStmt: AExprNT DOT IDENT LP OptArgListNT RP
MethodCallStmtNT |= (AExprNT, T.DOT, T.IDENT, T.LP, OptArgListNT, T.RP,
                     lambda objExpr, idTok, args: MethodCall(obj=objExpr, method=idTok, args=args))

# CloneCall: CLONE LP AExprNT COMMA AExprNT RP
CloneCallNT |= (T.CLONE, T.LP, AExprNT, OptCommaAExprNT, T.RP,
                lambda _clone, e1, e2: CloneCall(src=e1, dst=e2))

# AllocCall: ALLOC LP AExprNT OptCommaAExprNT RP
AllocCallNT |= (T.ALLOC, T.LP, AExprNT, OptCommaAExprNT, T.RP,
                lambda _alloc, expr, extra: AllocCall(expr=expr, extra=extra))

OptCommaAExprNT |= (T.COMMA, AExprNT, lambda expr: expr)
OptCommaAExprNT |= (Eps, lambda: None)

# FinalCall: FINAL LP AExprNT COMMA AExprNT RP
FinalCallNT |= (T.FINAL, T.LP, AExprNT, T.COMMA, AExprNT, T.RP,
                lambda _final, e1, e2: FinalCall(obj=e1, func=e2))

# OptArgListNT: опциональный список аргументов (для вызовов функций и методов)
OptArgListNT |= (ArgListNT, lambda lst: lst)
OptArgListNT |= (Eps, lambda: [])
ArgListNT |= (AExprNT, ArgListTailNT, lambda first, tail: [first] + tail)
ArgListNT |= (AExprNT, lambda first: [first])
ArgListTailNT |= (T.COMMA, AExprNT, ArgListTailNT, lambda comma, expr, tail: [expr] + tail)
ArgListTailNT |= (Eps, lambda: [])

# --- Выражения ---
# PrimaryAExprNT: базовые выражения – число, идентификатор, строка, скобки
PrimaryAExprNT |= (T.NUMBER, lambda numTok: AExprNum(value=int(numTok)))
PrimaryAExprNT |= (T.CHAR, lambda charTok: AExprChar(value=charTok))
PrimaryAExprNT |= (T.STRING, lambda sTok: AExprString(value=sTok))
PrimaryAExprNT |= (T.IDENT, lambda idTok: AExprVar(name=idTok))
PrimaryAExprNT |= (T.LP, AExprNT, T.RP, lambda expr: expr)

AssignExprNT |= (AExprAddChainNT, lambda expr: expr)
AssignExprNT |= (AExprAddChainNT, T.EQ, AssignExprNT,
                   lambda left, right: Assignment(left=left, right=right))

# PostfixChainNT: цепочка постфиксных операций – доступ к полям и индексирование
# Правило для доступа к полю: точка, идентификатор
# ТУТ НЕ ТОЧКА, А КВАДРАТНАЯ СКОБКА
PostfixTailItemNT |= (T.LSQUARE, T.IDENT,
                      lambda idTok: ("field", idTok))
# Правило для индексирования: "[", AExprNT, "]"
PostfixTailItemNT |= (T.LSQUARE, AExprNT, T.RSQUARE,
                      lambda expr: ("index", expr))

PostfixTailNT |= (PostfixTailItemNT, PostfixTailNT,
                  lambda item, tail: [item] + tail)
PostfixTailNT |= (Eps, lambda: [])

AExprPostfixNT |= (PrimaryAExprNT, PostfixTailNT,
                   lambda prim, tail: AExprPostfix(atom=prim, tail=tail))

# Правила для произведения (умножение и деление)
AExprMulChainNT |= (AExprPostfixNT, lambda expr: AExprMulChain(left=expr, tail=[]))
AExprMulChainNT |= (AExprMulChainNT, T.MUL, AExprPostfixNT,
                     lambda left, right: AExprMulChain(
                         left=left.left,
                         tail=left.tail + [("*", right)]
                     ))
AExprMulChainNT |= (AExprMulChainNT, T.DIV, AExprPostfixNT,
                   lambda left, right: AExprMulChain(
                       left=left.left,
                       tail=left.tail + [("/", right)]
                   ))

AExprMulChainNT |= (AExprMulChainNT, T.PERCENT, AExprPostfixNT,
                    lambda left, right: AExprMulChain(
                        left=left.left,
                        tail=left.tail + [("%", right)]
                    ))

# Правила для суммы (сложение и вычитание)
AExprAddChainNT |= (AExprMulChainNT, lambda expr: AExprAddChain(left=expr, tail=[]))
AExprAddChainNT |= (AExprAddChainNT, T.PLUS, AExprMulChainNT,
                   lambda left, right: AExprAddChain(
                       left=left.left,
                       tail=left.tail + [("+", right)]
                   ))
AExprAddChainNT |= (AExprAddChainNT, T.MINUS, AExprMulChainNT,
                   lambda left, right: AExprAddChain(
                       left=left.left,
                       tail=left.tail + [("-", right)]
                   ))

AExprNT |= (AExprPostfixNT, lambda expr: expr)
AExprNT |= (AExprAddChainNT, lambda expr: expr)
#AExprNT |= (Eps, lambda: "")

# Boolean expression (для условий if, while).
BExprNT |= (AExprNT, lambda expr: expr)
BExprNT |= (AExprNT, T.LT, AExprNT,
             lambda left, right: Comparison(op="<", left=left, right=right))
BExprNT |= (AExprNT, T.GT, AExprNT,
             lambda left, right: Comparison(op=">", left=left, right=right))
BExprNT |= (AExprNT, T.LE, AExprNT,
             lambda left, right: Comparison(op="<=", left=left, right=right))
BExprNT |= (AExprNT, T.GE, AExprNT,
             lambda left, right: Comparison(op=">=", left=left, right=right))
BExprNT |= (AExprNT, T.EQEQ, AExprNT,
             lambda left, right: Comparison(op="==", left=left, right=right))
BExprNT |= (AExprNT, T.NEQ, AExprNT,
             lambda left, right: Comparison(op="!=", left=left, right=right))
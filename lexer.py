#!/usr/bin/env python3

import sys
import re
import enum
sys.path.append('..')

from parser_edsl_new import Symbol, Terminal, SpecTerminal

def spec_terminal_match(self, string, pos):
    if string.startswith(self.name, pos):
        return len(self.name), None
    else:
        return 0, None

SpecTerminal.match = spec_terminal_match

class T:
    # Ключевые слова
    FUNC    = Terminal("func",   r"func",   lambda s: None, priority=10, re_flags=re.IGNORECASE)
    TYPE    = Terminal("type",   r"type",   lambda s: s, priority=10, re_flags=re.IGNORECASE)
    CONST   = Terminal("const",  r"const",  lambda s: s, priority=10, re_flags=re.IGNORECASE)
    VAR     = Terminal("var",    r"var",    lambda s: s, priority=10, re_flags=re.IGNORECASE)
    CLASS   = Terminal("class",  r"class",  lambda s: s, priority=10, re_flags=re.IGNORECASE)
    DYNVAR  = Terminal("dynvar", r"dynvar", lambda s: s, priority=10, re_flags=re.IGNORECASE)
    PFIELD  = Terminal("pfield", r"pfield", lambda s: s, priority=10, re_flags=re.IGNORECASE)
    IF      = Terminal("if",     r"if",     lambda s: s, priority=10, re_flags=re.IGNORECASE)
    ELSE    = Terminal("else",   r"else",   lambda s: s, priority=10, re_flags=re.IGNORECASE)
    WHILE   = Terminal("while",  r"while",  lambda s: s, priority=10, re_flags=re.IGNORECASE)
    RETURN  = Terminal("return", r"return", lambda s: s, priority=10, re_flags=re.IGNORECASE)
    CLONE   = Terminal("clone",  r"clone",  lambda s: s, priority=10, re_flags=re.IGNORECASE)
    ALLOC   = Terminal("alloc",  r"alloc",  lambda s: s, priority=10, re_flags=re.IGNORECASE)
    FINAL   = Terminal("final",  r"final",  lambda s: s, priority=10, re_flags=re.IGNORECASE)

    # Разделители и операторы
    LBRACE      = SpecTerminal("{")
    RBRACE      = SpecTerminal("}")
    LSQUARE     = SpecTerminal("[")
    RSQUARE     = SpecTerminal("]")
    LP          = SpecTerminal("(")
    RP          = SpecTerminal(")")
    COMMA       = SpecTerminal(",")
    DOT         = SpecTerminal(".")
    EQ          = SpecTerminal("=")
    COLON       = SpecTerminal(":")
    SEMICOLON   = SpecTerminal(";")
    PLUS        = SpecTerminal("+")
    MINUS       = SpecTerminal("-")
    MUL         = SpecTerminal("*")
    DIV         = SpecTerminal("/")
    PERCENT     = SpecTerminal("%")
    SYM         = SpecTerminal("[]")
    ANDAND      = SpecTerminal("&&")
    OROR        = SpecTerminal("||")
    EQEQ        = SpecTerminal("==")
    NEQ         = SpecTerminal("!=")
    LT          = SpecTerminal("<")
    LE          = SpecTerminal("<=")
    GT          = SpecTerminal(">")
    GE          = SpecTerminal(">=")
    AMP         = SpecTerminal("&")
    MULD        = SpecTerminal("*D")
    DOT_LPAREN  = SpecTerminal(".(")

    # Лексемы, для которых используются регулярные выражения
    INT     = Terminal("int",  r"int",  lambda s: s)
    CHAR_KW = Terminal("char", r"char", lambda s: s)

    # Токены для идентификаторов, чисел, строк, символов и булевых значений
    IDENT  = Terminal("IDENT",  r"[A-Za-z][A-Za-z0-9_]*", lambda s: s)
    NUMBER = Terminal("NUMBER", r"[0-9]+", int)
    STRING = Terminal("STRING", r"\"(\\.|[^\"])*\"", lambda s: s[1:-1])
    CHAR   = Terminal("CHAR",   r"'(\\.|[^'])'", lambda s: s[1:-1])
    TRUE   = Terminal("true",   r"true",  lambda s: True,  re_flags=re.IGNORECASE)
    FALSE  = Terminal("false",  r"false", lambda s: False, re_flags=re.IGNORECASE)
    NIL    = Terminal("nil",    r"nil",   lambda s: None,  re_flags=re.IGNORECASE)

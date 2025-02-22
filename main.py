#!/usr/bin/env python3
import abc
import sys
import traceback
sys.path.append('..')

from grammar import *

# ============================================================================
# 3. Функция запуска парсера и вывод AST
# ============================================================================

def main():
    # Изменённый метод Program.check, чтобы принимать symbols по умолчанию
    def program_check(self, symbols=None):
        if symbols is None:
            symbols = {}
        for decl in self.top_levels:
            decl.check(symbols)

    # Подменяем метод check для Program, если нужно:
    Program.check = program_check

    samples = [
        """
        type MyDyn dynvar {
        ptr *D
        count int
        }
    """
    ]

    for idx, sample in enumerate(samples, 1):
        print(f"=== Пример {idx} ===")
        print("Исходный код:")
        print(sample)
        try:
            # Токенизация
            domains = []
            for obj in T.__dict__.values():
                if isinstance(obj, (Terminal, SpecTerminal)):
                    if isinstance(obj, SpecTerminal) and not hasattr(obj, 'priority'):
                        obj.priority = 10
                    domains.append(obj)
            lexer = pe.Lexer(domains=domains, text=sample, skip=[r'\s+', r'\{.*?\}'])
            tokens = []
            while True:
                token = lexer.next_token()
                tokens.append(token)
                if token.type == pe.EOF_SYMBOL:
                    break
            print("Tokens:")
            for token in tokens:
                print(token.pos, ":", token)

            # Парсинг
            p = pe.Parser(ProgramStart)
            p.add_skipped_domain(r'\s+')
            p.add_skipped_domain(r'\{.*?\}')
            ast = p.parse(sample)
            print("\nParsed AST:")
            from pprint import pprint
            pprint(ast)

            ast.check()
            print("Semantic analysis passed successfully.")
        except Exception as e:
            print("Error encountered:")
            traceback.print_exc()
            try:
                p.print_table()
            except Exception:
                pass
        print("\n" + "=" * 40 + "\n")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import abc
import sys
import traceback
import io
sys.path.append('..')

from grammar import *

def main():
    def program_check(self, symbols=None):
        if symbols is None:
            symbols = {}
        for decl in self.top_levels:
            decl.check(symbols)

    Program.check = program_check

    samples = [
        """
        func gcd(a int, b int) int {
        var rem int
        while b != 0 {
            rem = a % b
            a = b
            b = rem
        }
        return a
    }
    """,
        """
            func gcd(a int, b str) int {
            var rem int
            while b != 0 {
                 rem = a % b
                   a = b
                  b = rem
              }
             return a
         }
        """,
        """
                func main() int {
                   var x int
                      x = 10
                      return x
                }
        """,
        """
            type MyStruct {
                  x int
                  y [10]char
            }
        """
    ]

    for idx, sample in enumerate(samples, 1):
        print(f"=== Пример {idx} ===")
        print("Исходный код:")
        print(sample)
        try:
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

            p = pe.Parser(ProgramStart)
            p.add_skipped_domain(r'\s+')
            p.add_skipped_domain(r'\{.*?\}')
            ast = p.parse(sample)
            print("\nParsed AST:")
            from pprint import pprint
            pprint(ast)

            # Перенаправляем вывод для захвата семантических ошибок
            error_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = error_capture
            ast.check()
            sys.stdout = old_stdout
            semantic_errors = error_capture.getvalue().strip()
            if semantic_errors:
                print("Semantic Errors:")
                print(semantic_errors)
            else:
                print("Semantic analysis passed successfully.")

            code = ast.generate()
            print("НУИЯП! Во имя РАЯП!:\n", code)
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

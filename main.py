#!/usr/bin/env python3
import sys
import traceback
import argparse
import io
from pprint import pformat
from grammar import *  # импортирует AST, парсер и т.д.
import parser_edsl_new as pe

def main():
    parser = argparse.ArgumentParser(description="Компиляция исходного файла Go-подобного синтаксиса в AST, семантический анализ и NUYAP представление.")
    parser.add_argument("source", help="Входной файл с исходным кодом (Go-подобный синтаксис)")
    parser.add_argument("-o", "--output", help="Выходной файл, куда записывается результат", default=None)
    args = parser.parse_args()

    try:
        with open(args.source, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        print(f"Ошибка при чтении файла {args.source}: {e}")
        sys.exit(1)

    domains = []
    for obj in T.__dict__.values():
        if isinstance(obj, (Terminal, SpecTerminal)):
            if isinstance(obj, SpecTerminal) and not hasattr(obj, 'priority'):
                obj.priority = 10
            domains.append(obj)
    lexer = pe.Lexer(domains=domains, text=source_code, skip=[r'\s+', r'\{.*?\}'])
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

    try:
        ast = p.parse(source_code)
    except pe.Error as e:
        print(f"Ошибка парсинга {e.pos}: {e.message}")
        sys.exit(1)
    except Exception as ex:
        traceback.print_exc()
        sys.exit(1)

    semantic_output = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = semantic_output
        ast.check()
    finally:
        sys.stdout = old_stdout
    semantic_errors = semantic_output.getvalue().strip()

    ast_tree_str = pformat(ast)

    nuyap_code = ast.generate()

    output_text = []
    output_text.append("=== AST Tree ===")
    output_text.append(ast_tree_str)
    output_text.append("\n=== Semantic Analysis ===")
    output_text.append(semantic_errors if semantic_errors else "No semantic errors.")
    output_text.append("\n=== НУИЯП ===")
    output_text.append(nuyap_code)
    final_output = "\n".join(output_text)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as out_file:
                out_file.write(final_output)
            print(f"Результат записан в {args.output}")
        except Exception as e:
            print(f"Ошибка записи в файл {args.output}: {e}")
            sys.exit(1)
    else:
        print(final_output)

if __name__ == "__main__":
    main()
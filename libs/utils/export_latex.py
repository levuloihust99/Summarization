import csv
import argparse
import unicodedata
from typing import Text

from libs.text_processing.unicode_util import Zs_category


def latex_escape(text: Text):
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\textbackslash")
        elif ch == "~":
            out.append("\\textasciitilde")
        elif ch == "^":
            out.append("\\textasciicircum")
        elif ch in {
            "%", "$", "&", "#", "^", "_", "}", "{"
        }:
            out.append("\\{}".format(ch))
        elif ch in Zs_category:
            out.append(" ")
        elif ch == "\u200b":
            pass
        else:
            out.append(ch)
    return "".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--output_path", required=True,
                        help="Path to the output LaTeX file.")
    parser.add_argument("--n_columns", type=int, default=4)
    parser.add_argument("--width", type=float, default=4.0)
    parser.add_argument("--layout", type=eval, default=[0.01, 0.6, 0.1, 0.1, 0.1])
    parser.add_argument("--template_file", default="latex/template.txt")
    parser.add_argument("--add_order", action="store_true", default=False)
    args = parser.parse_args()

    row_count = 0
    with open(args.input_path, "r") as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        n_columns = len(header)
        if args.add_order:
            n_columns += 1
            header = ["ID"] + header
        assert len(args.layout) == n_columns

        if not args.add_order:
            layout_latex = "p{{{}\linewidth}}" * n_columns
        else:
            layout_latex = "P{{{}\linewidth}}" + "p{{{}\linewidth}}" * (n_columns - 1)
        layout_latex = layout_latex.format(*args.layout)

        content = ""
        if not args.add_order: 
            header_latex = " & ".join([r"\multicolumn{{1}}{{c}}{{\textbf{{{}}}}}"] * n_columns)
        else:
            header_latex = " & ".join([r"\textbf{{{}}}"] + [r"\multicolumn{{1}}{{c}}{{\textbf{{{}}}}}"] * (n_columns - 1))
        header_latex = header_latex.format(*header)
        header_latex = "\\rowcolor{light-gray}" + header_latex
        content += header_latex + "\\\\\n"
        for row in csv_reader:
            row = [unicodedata.normalize("NFKC", cell) for cell in row]
            if args.add_order:
                row = [str(row_count + 1)] + row
            row_latex = " & ".join(["{}"] * n_columns)
            row_latex = row_latex.format(*[latex_escape(cell) for cell in row])
            row_latex += "\\\\\n"
            if row_count % 2 == 1:
                row_latex = "\\rowcolor{light-gray}" + row_latex
            content += row_latex
            row_count += 1
    
    with open(args.template_file, "r") as reader:
        template = reader.read()
    output = template.format(width=args.width, layout=layout_latex, content=content)
    
    with open(args.output_path, "w") as writer:
        writer.write(output)


if __name__ == "__main__":
    main()

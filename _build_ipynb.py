"""Convierte goaloracle_notebook.py (formato '# %%') a un .ipynb válido para Colab."""
import json
import sys

SRC = "goaloracle_notebook.py"
OUT = "GoalOracle_Prediccion.ipynb"


def parse_cells(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    cells, cur = [], None
    for line in lines:
        if line.startswith("# %%"):
            if cur is not None:
                cells.append(cur)
            ctype = "markdown" if "[markdown]" in line else "code"
            cur = {"type": ctype, "lines": []}
        else:
            if cur is None:  # preámbulo antes del primer marcador
                cur = {"type": "code", "lines": []}
            cur["lines"].append(line)
    if cur is not None:
        cells.append(cur)
    return cells


def strip_blanks(lines):
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


def build_notebook(cells):
    nb_cells = []
    for c in cells:
        src = strip_blanks(list(c["lines"]))
        if not src:
            continue
        if c["type"] == "markdown":
            # quita el prefijo de comentario '# ' / '#'
            src = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l)
                   for l in src]
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": "\n".join(src)})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {},
                             "execution_count": None, "outputs": [],
                             "source": "\n".join(src)})

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_notebook(parse_cells(SRC))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"OK -> {OUT} ({len(nb['cells'])} celdas)")
    sys.exit(0)

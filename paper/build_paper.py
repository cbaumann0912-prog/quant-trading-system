import pathlib
import re
import subprocess

SRC = pathlib.Path("paper_new.md")
raw = SRC.read_text(encoding="utf-8")
lines = raw.split("\n")
assert lines[0].startswith("# Underpowered by Design")
body = "\n".join(lines[1:]).lstrip("\n")

META = """---
title: "Underpowered by Design"
subtitle: "Six Pre-Registered Null Results in Systematic FX Research"
author: "Clay Baumann"
date: "Cornell Engineering, August 2026"
{extra}
---

"""

PDF_EXTRA = """toc: true
toc-depth: 2
numbersections: false
geometry: "margin=1.05in"
fontsize: 11pt
linestretch: 1.06
colorlinks: true
linkcolor: "black"
urlcolor: "MidnightBlue\""""


def contents_block(text):
    entries = []
    for m in re.finditer(r"(?m)^(#{1,2}) (.+)$", text):
        level, title = len(m.group(1)), m.group(2).strip()
        if level == 1:
            entries.append(("H", title))
        else:
            entries.append(("S", title))
    out = ["## Contents", ""]
    for i, (kind, title) in enumerate(entries):
        last = i == len(entries) - 1
        brk = "" if last else "\\"
        if kind == "H":
            if i:
                out.append("")
            out.append(f"**{title}**{brk}")
        else:
            out.append(f"{title}{brk}")
    inner = "\n".join(out[2:])
    return ("## Contents\n\n"
            '::: {custom-style="TOCEntry"}\n'
            + inner + "\n:::\n\n---\n")


pathlib.Path("build_pdf.md").write_text(
    META.format(extra=PDF_EXTRA) + body, encoding="utf-8")

def number_figures(text):
    n = [0]

    def repl(m):
        n[0] += 1
        return f"![**Figure {n[0]}.** {m.group(1)}"

    return re.sub(r"!\[(?!\*\*Figure )", lambda m: repl(type("M", (), {"group": lambda s, i: ""})()), text) \
        if False else re.sub(r"!\[", lambda m: (n.__setitem__(0, n[0] + 1), f"![**Figure {n[0]}.** ")[1], text)


docx_body = contents_block(body) + "\n" + number_figures(body)
pathlib.Path("build_docx.md").write_text(
    META.format(extra='') + docx_body, encoding="utf-8")

subprocess.run(["pandoc", "build_pdf.md", "-o", "paper.tex",
                "--standalone", "--pdf-engine=xelatex"], check=True)
tex = pathlib.Path("paper.tex").read_text(encoding="utf-8")
pathlib.Path("paper.tex").write_text(
    tex.replace("\\usepackage{lmodern}\n", ""), encoding="utf-8")
for _ in range(2):
    subprocess.run(["xelatex", "-interaction=nonstopmode", "paper.tex"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.run(["pandoc", "build_docx.md",
                "-o", "Underpowered by Design - Baumann.docx",
                "--reference-doc=reference.docx", "--standalone"], check=True)
subprocess.run(["cp", "paper.pdf", "Underpowered by Design - Baumann.pdf"],
               check=True)
print("built")

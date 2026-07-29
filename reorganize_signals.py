"""One-shot reorganisation of src/signals into primitives/ and strategies/.

Run once from the repository root, then delete this file:

    python reorganize_signals.py --dry-run    # see what would change
    python reorganize_signals.py              # do it
    python -m pytest -p no:cacheprovider      # verify

Moves are done with `git mv` when the repo is clean enough for it, so history
follows the files. Imports are rewritten across src/, tests/ and research/.
Nothing is deleted; the only destructive operation is the move itself.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SIGNALS = REPO / "src" / "signals"

PRIMITIVES = [
    "momentum",
    "mean_reversion",
    "cointegration",
    "triple_barrier",
    "regime_gated",
    "regime_refit",
]

STRATEGIES = [
    "pc2_carry",
    "momentum_ml_regime",
    "ou_reversion",
    "month_end_flow",
    "intraday_overshoot",
]

# signal_builder stays at src/signals/ — it is the engine, not a signal.

SEARCH_ROOTS = ["src", "tests", "research"]

PACKAGE_DOCSTRING = {
    "primitives": (
        '"""Reusable signal primitives.\n\n'
        "Generic constructions not tied to any one hypothesis: momentum,\n"
        "mean reversion, cointegration, labelling, and regime gating. Strategy\n"
        "modules compose these; strategies 4 and 4b are built from these alone.\n"
        '"""\n'
    ),
    "strategies": (
        '"""Signal construction for individual pre-registered strategies.\n\n'
        "One module per strategy that required bespoke construction. Each was\n"
        "extracted verbatim from the validation script that produced its\n"
        "verdict, and verified to reproduce the recorded figures.\n"
        '"""\n'
    ),
}


def run_git(args: list[str]) -> bool:
    try:
        subprocess.run(["git", *args], cwd=REPO, check=True,
                       capture_output=True, text=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def move(src: Path, dst: Path, dry: bool) -> str:
    rel_src = src.relative_to(REPO).as_posix()
    rel_dst = dst.relative_to(REPO).as_posix()
    if dry:
        return f"  would move  {rel_src} -> {rel_dst}"
    if run_git(["mv", rel_src, rel_dst]):
        return f"  git mv      {rel_src} -> {rel_dst}"
    src.replace(dst)
    return f"  moved       {rel_src} -> {rel_dst}  (plain move; git will detect the rename)"


def build_rewrites() -> list[tuple[re.Pattern[str], str]]:
    rules = []
    for group, modules in (("primitives", PRIMITIVES), ("strategies", STRATEGIES)):
        for mod in modules:
            rules.append((
                re.compile(rf"(?<![\w.])src\.signals\.{mod}(?![\w])"),
                f"src.signals.{group}.{mod}",
            ))
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry = args.dry_run

    if not SIGNALS.is_dir():
        print(f"error: {SIGNALS} not found — run from the repository root")
        return 1

    missing = [m for m in PRIMITIVES + STRATEGIES if not (SIGNALS / f"{m}.py").exists()]
    already = [m for m in PRIMITIVES if (SIGNALS / "primitives" / f"{m}.py").exists()]
    if already:
        print(f"error: looks like this already ran ({already[0]} is in primitives/)")
        return 1
    if missing:
        print(f"error: expected modules not found in src/signals: {missing}")
        return 1

    print("packages")
    for group in ("primitives", "strategies"):
        pkg = SIGNALS / group
        init = pkg / "__init__.py"
        if dry:
            print(f"  would create {pkg.relative_to(REPO).as_posix()}/__init__.py")
        else:
            pkg.mkdir(exist_ok=True)
            init.write_text(PACKAGE_DOCSTRING[group], encoding="utf-8")
            print(f"  created     {init.relative_to(REPO).as_posix()}")

    print("\nmoves")
    for group, modules in (("primitives", PRIMITIVES), ("strategies", STRATEGIES)):
        for mod in modules:
            print(move(SIGNALS / f"{mod}.py", SIGNALS / group / f"{mod}.py", dry))

    print("\nimport rewrites")
    rules = build_rewrites()
    touched = 0
    for root in SEARCH_ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            if "__pycache__" in path.parts or ".venv" in path.parts:
                continue
            original = path.read_text(encoding="utf-8")
            updated = original
            for pattern, replacement in rules:
                updated = pattern.sub(replacement, updated)
            if updated != original:
                touched += 1
                n = sum(1 for a, b in zip(original.splitlines(), updated.splitlines()) if a != b)
                print(f"  {'would update' if dry else 'updated'}  "
                      f"{path.relative_to(REPO).as_posix()}  ({n} line(s))")
                if not dry:
                    path.write_text(updated, encoding="utf-8")

    print(f"\n{touched} file(s) {'would be ' if dry else ''}rewritten")
    if not dry:
        print("\nnext:")
        print("  python -m pytest -p no:cacheprovider")
        print("  then delete this script")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the English and Simplified Chinese documentation.

``python docs/build.py`` checks that every public name has an API entry and
that the zh_CN catalog covers every source string, then writes both editions
and a language chooser to docs/_build/html. It also lists the README rows and
user guide cards that are still missing; these are filled in when a group of
new modules is integrated.
``python docs/build.py --update-po`` merges new and changed source strings
into the catalog (changed ones are marked fuzzy) and stops.
``python docs/build.py --merge-po`` resolves a merge or rebase conflict in the
catalog by keeping the translations of both sides, then updates it like
``--update-po``.
``python docs/build.py --release VERSION`` moves the release-note fragments
into a section for VERSION dated today and deletes them. It refuses while any
README row or user guide card is missing.
"""

import argparse
import ast
import datetime
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import _fragments
from babel.messages.pofile import read_po, write_po

docs = Path(__file__).resolve().parent
build = docs / "_build"
catalog_path = docs / "locale" / "zh_CN" / "LC_MESSAGES" / "kivi.po"
editions = (("en", "en"), ("zh", "zh_CN"))
package = docs.parent / "src" / "kivi"
readmes = ("README.md", "README.zh-CN.md")
# Public names that the README tables leave out on purpose. Exceptions are left out too.
not_in_readme = {"kivi.datasets.Dataset": "the 0.2.x loader, kept for old code; see load_bank"}
api_entry = re.compile(
    r"^\s*\.\. auto(?:function|class|exception|method|data|attribute)::\s+([\w.]+)", re.MULTILINE
)


def sphinx(builder: str, output: Path, *options: str) -> None:
    doctrees = build / "doctrees" / output.name
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "-q",
            "-b",
            builder,
            "-d",
            str(doctrees),
            *options,
            str(docs),
            str(output),
        ],
        check=True,
    )


def extract_template():
    sphinx("gettext", build / "gettext")
    with open(build / "gettext" / "kivi.pot", "rb") as source:
        return read_po(source)


def write_catalog(catalog) -> None:
    with open(catalog_path, "wb") as target:
        write_po(
            target,
            catalog,
            width=100,
            no_location=True,
            ignore_obsolete=True,
            include_previous=False,
        )


def update_catalog(template) -> None:
    with open(catalog_path, "rb") as source:
        catalog = read_po(source, locale="zh_CN")
    created = catalog.creation_date
    catalog.update(template, update_header_comment=False)
    # A fresh POT-Creation-Date in every pull request would make all of them conflict.
    catalog.creation_date = created
    write_catalog(catalog)
    pending = [m.id for m in catalog if m.id and (m.fuzzy or not m.string)]
    print(f"Updated {catalog_path.relative_to(docs.parent)}; {len(pending)} entries to translate.")


def merge_catalog() -> None:
    """Union the two conflicting versions of the catalog, preferring settled translations."""
    relative = catalog_path.relative_to(docs.parent).as_posix()
    sides = []
    for stage in (2, 3):  # 2 is the checked-out side, 3 the side being merged in
        shown = subprocess.run(
            ["git", "show", f":{stage}:{relative}"], cwd=docs.parent, capture_output=True
        )
        if shown.returncode:
            sys.exit(f"{relative} has no merge conflict to resolve.")
        sides.append(read_po(io.BytesIO(shown.stdout), locale="zh_CN"))
    catalog, other = sides
    clashes = []
    for message in other:
        if not message.id:
            continue
        entry = catalog.get(message.id, message.context)
        settled = bool(message.string) and not message.fuzzy
        if entry is None:
            catalog[message.id] = message
        elif settled and (not entry.string or entry.fuzzy):
            entry.string = message.string
            entry.flags.discard("fuzzy")
        elif settled and entry.string != message.string:
            clashes.append(message.id)
    write_catalog(catalog)
    for text in clashes:
        print(f"  both sides translate differently, kept the checked-out one: {text[:80]!r}")
    update_catalog(extract_template())
    print(f"Translate any entries reported above, then `git add {relative}`.")


def release_notes(version: str) -> None:
    """Move the release-note fragments into a dated section at the top of the notes."""
    found = _fragments.collect()
    paths = [path for group in found.values() for path in group]
    if not paths:
        sys.exit("No release-note fragments in docs/release/unreleased.")
    notes = docs / "release" / "index.rst"
    text = notes.read_text(encoding="utf-8")
    marker = ".. release-fragments::\n\n"
    if text.count(marker) != 1:
        sys.exit(f"Expected one release-fragments directive in {notes.name}.")
    heading = f"{version} ({datetime.date.today().isoformat()})"
    section = f"{heading}\n{'-' * len(heading)}\n\n{_fragments.text(found)}\n"
    notes.write_text(text.replace(marker, marker + section), encoding="utf-8")
    for path in paths:
        path.unlink()
    print(
        f"Added {heading} to {notes.relative_to(docs.parent)} from {len(paths)} fragments. "
        f"Run `python docs/build.py --update-po` and translate the new heading."
    )


def public_api() -> dict[str, list[str]]:
    """Map each public module of kivi to the names in its ``__all__``.

    The source is parsed rather than imported, so that the check always reads
    this checkout even when an editable install points at another one.
    """
    modules = {}
    sources = [(path.stem, path) for path in sorted(package.glob("*.py"))]
    sources += [(path.parent.name, path) for path in sorted(package.glob("*/__init__.py"))]
    for module, path in sources:
        if module.startswith("_"):
            continue
        names = None
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                targets = [node.target]
            else:
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in targets
            ):
                continue
            try:
                value = list(ast.literal_eval(node.value))
            except ValueError:
                sys.exit(f"kivi.{module} must give __all__ as a literal list of names.")
            names = names + value if isinstance(node, ast.AugAssign) and names else value
        if names is None:
            sys.exit(f"kivi.{module} must list its public names in __all__.")
        modules[module] = names
    return modules


def check_api_pages(modules: dict[str, list[str]]) -> None:
    """Stop unless docs/api/<module>.rst documents every public name of the module."""
    problems = []
    for module, names in modules.items():
        page = docs / "api" / f"{module}.rst"
        if not page.exists():
            problems.append(f"kivi.{module} has no API page docs/api/{module}.rst")
            continue
        # ``automethod:: Dataset.bank_data`` documents the class Dataset through its method.
        documented = {
            target.removeprefix(f"kivi.{module}.").split(".")[0]
            for target in api_entry.findall(page.read_text(encoding="utf-8"))
        }
        problems += [
            f"docs/api/{module}.rst has no entry for {name}"
            for name in names
            if name not in documented
        ]
    if problems:
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        sys.exit(f"{len(problems)} public names without an API entry.")


def entry_point_gaps(modules: dict[str, list[str]]) -> list[str]:
    """List the README rows and user guide cards that the integration still has to add."""
    guides = sorted(p.stem for p in (docs / "user_guide").glob("*.rst") if p.stem != "index")
    gaps = []
    for readme in readmes:
        text = (docs.parent / readme).read_text(encoding="utf-8")
        rows = "\n".join(line for line in text.splitlines() if line.startswith("|"))
        for module, names in modules.items():
            for name in names:
                listed = re.search(rf"`kivi\.{module}\.{name}[`.(]", rows)
                exempt = name.endswith("Error") or f"kivi.{module}.{name}" in not_in_readme
                if not (listed or exempt):
                    gaps.append(f"{readme} does not list kivi.{module}.{name}")
        for page in guides:
            for edition in ("en", "zh"):
                if f"/{edition}/user_guide/{page}.html" not in text:
                    gaps.append(f"{readme} does not link the {edition} edition of {page}")
    cards = (docs / "user_guide" / "index.rst").read_text(encoding="utf-8")
    gaps += [
        f"docs/user_guide/index.rst has no card for {page}"
        for page in guides
        if not re.search(rf"^\s+:link: {page}\s*$", cards, re.MULTILINE)
    ]
    return gaps


def report_gaps(gaps: list[str]) -> None:
    print(f"{len(gaps)} entry points to update before the next release:", file=sys.stderr)
    for gap in gaps:
        print(f"  {gap}", file=sys.stderr)


def check_catalog(template) -> None:
    with open(catalog_path, "rb") as source:
        catalog = read_po(source, locale="zh_CN")
    problems = []
    for message in template:
        if not message.id:
            continue
        entry = catalog.get(message.id, message.context)
        if entry is None or not entry.string:
            problems.append(("untranslated", message.id))
        elif entry.fuzzy:
            problems.append(("fuzzy", message.id))
    expected = {(m.id, m.context) for m in template}
    problems += [("stale", m.id) for m in catalog if m.id and (m.id, m.context) not in expected]
    if problems:
        for kind, text in problems:
            print(f"{kind:>12}: {text[:100]!r}", file=sys.stderr)
        sys.exit(
            f"{len(problems)} catalog problems. Run `python docs/build.py --update-po`, "
            f"then translate the empty and fuzzy entries in {catalog_path.name}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--update-po",
        action="store_true",
        help="merge source strings into the zh_CN catalog and stop",
    )
    action.add_argument(
        "--merge-po",
        action="store_true",
        help="resolve a merge conflict in the zh_CN catalog and stop",
    )
    action.add_argument(
        "--release",
        metavar="VERSION",
        help="move the release-note fragments into a dated section and stop",
    )
    args = parser.parse_args()
    if args.merge_po:
        merge_catalog()
        return
    if args.update_po:
        update_catalog(extract_template())
        return
    modules = public_api()
    check_api_pages(modules)
    gaps = entry_point_gaps(modules)
    if gaps:
        report_gaps(gaps)
    if args.release:
        if gaps:
            sys.exit("Update them first, as docs/development/documentation.rst describes.")
        release_notes(args.release)
        return
    check_catalog(extract_template())
    html = build / "html"
    shutil.rmtree(html, ignore_errors=True)
    for folder, locale in editions:
        sphinx("html", html / folder, "-D", f"language={locale}")
    shutil.copyfile(docs / "landing.html", html / "index.html")
    (html / ".nojekyll").touch()


if __name__ == "__main__":
    main()

"""Build the English and Simplified Chinese documentation.

``python docs/build.py`` checks that the zh_CN catalog covers every source
string, then writes both editions and a language chooser to docs/_build/html.
``python docs/build.py --update-po`` merges new and changed source strings
into the catalog (changed ones are marked fuzzy) and stops.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from babel.messages.pofile import read_po, write_po

docs = Path(__file__).resolve().parent
build = docs / "_build"
catalog_path = docs / "locale" / "zh_CN" / "LC_MESSAGES" / "kivi.po"
editions = (("en", "en"), ("zh", "zh_CN"))


def sphinx(builder: str, output: Path, *options: str) -> None:
    doctrees = build / "doctrees" / output.name
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-b", builder, "-d", str(doctrees),
         *options, str(docs), str(output)],
        check=True,
    )


def extract_template():
    sphinx("gettext", build / "gettext")
    with open(build / "gettext" / "kivi.pot", "rb") as source:
        return read_po(source)


def update_catalog(template) -> None:
    with open(catalog_path, "rb") as source:
        catalog = read_po(source, locale="zh_CN")
    catalog.update(template, update_header_comment=False)
    with open(catalog_path, "wb") as target:
        write_po(target, catalog, width=100, no_location=True, ignore_obsolete=True,
                 include_previous=False)
    pending = [m.id for m in catalog if m.id and (m.fuzzy or not m.string)]
    print(f"Updated {catalog_path.relative_to(docs.parent)}; {len(pending)} entries to translate.")


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
        sys.exit(f"{len(problems)} catalog problems. Run `python docs/build.py --update-po`, "
                 f"then translate the empty and fuzzy entries in {catalog_path.name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-po", action="store_true",
                        help="merge source strings into the zh_CN catalog and stop")
    args = parser.parse_args()
    template = extract_template()
    if args.update_po:
        update_catalog(template)
        return
    check_catalog(template)
    html = build / "html"
    shutil.rmtree(html, ignore_errors=True)
    for folder, locale in editions:
        sphinx("html", html / folder, "-D", f"language={locale}")
    shutil.copyfile(docs / "landing.html", html / "index.html")
    (html / ".nojekyll").touch()


if __name__ == "__main__":
    main()

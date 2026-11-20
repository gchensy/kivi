"""Release notes assembled from one fragment file per change.

A pull request describes its user-facing changes in
``docs/release/unreleased/<topic>.<category>.rst`` instead of editing
``docs/release/index.rst``, so two pull requests never change the same lines.
A fragment holds one or more bullet items. The ``release-fragments`` directive
renders the fragments under a heading per category, and
``python docs/build.py --release VERSION`` moves them into a dated section.
"""

from pathlib import Path

from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

folder = Path(__file__).resolve().parent / "release" / "unreleased"
CATEGORIES = {"new": "New", "changed": "Changed", "fixed": "Fixed", "removed": "Removed"}


def collect() -> dict[str, list[Path]]:
    """Fragment files by category, each list sorted by file name."""
    found = {category: [] for category in CATEGORIES}
    for path in sorted(folder.glob("*.rst")):
        topic, _, category = path.stem.rpartition(".")
        if not topic or category not in found:
            raise ValueError(
                f"{path.name}: name release-note fragments <topic>.<{'|'.join(CATEGORIES)}>.rst"
            )
        found[category].append(path)
    return found


def lines(found: dict[str, list[Path]]) -> list[tuple[str, Path | None, int]]:
    """RST lines of the fragments with category headings, and where each line comes from."""
    result = []
    for category, paths in found.items():
        if not paths:
            continue
        title = CATEGORIES[category]
        result += [(title, None, 0), ("^" * len(title), None, 0), ("", None, 0)]
        for path in paths:
            text = path.read_text(encoding="utf-8").strip("\n").splitlines()
            result += [(line, path, number) for number, line in enumerate(text)]
            result.append(("", path, len(text)))
    return result


def text(found: dict[str, list[Path]]) -> str:
    """The fragments as one RST block, as written into a dated section."""
    return "\n".join(line for line, _, _ in lines(found)).strip("\n") + "\n"


class ReleaseFragments(SphinxDirective):
    """Render the unreleased fragments; each line keeps its file so it can be translated."""

    def run(self):
        found = collect()
        source, line = self.get_source_info()
        content = StringList()
        for value, path, number in lines(found):
            if path is None:
                content.append(value, source, line)
            else:
                self.env.note_dependency(str(path))
                content.append(value, str(path), number)
        return self.parse_text_to_nodes(content, allow_section_headings=True)


def _always_reread(app, env, added, changed, removed):
    # A new fragment file is not a dependency yet, so re-read the page on every build.
    return ["release/index"]


def setup(app):
    app.add_directive("release-fragments", ReleaseFragments)
    app.connect("env-get-outdated", _always_reread)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

"""Build both languages with separate search indices and a shared entry point."""

from pathlib import Path
import subprocess
import sys

docs = Path(__file__).resolve().parent
output = docs / "_build" / "html"
for folder, locale in (("en", "en"), ("zh", "zh_CN")):
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-b", "html",
         "-D", f"language={locale}", str(docs), str(output / folder)],
        check=True,
    )
(output / "index.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8">'
    '<title>kivi documentation · 使用文档</title><main><h1>kivi</h1>'
    '<p><a lang="en" href="en/index.html">English documentation</a></p>'
    '<p><a lang="zh-CN" href="zh/index.html">中文文档</a></p></main></html>\n',
    encoding="utf-8",
)

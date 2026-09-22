"""Bundle the local Tone Lab assets into one offline HTML file (stdlib only)."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ("engine.js", "state.js", "app.js", "lightbox.js")


def build() -> str:
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    css_tag = '<link rel="stylesheet" href="style.css">'
    script_tags = "".join(f'<script src="{name}"></script>\n' for name in SCRIPTS)
    if html.count(css_tag) != 1 or html.count(script_tags) != 1:
        raise ValueError("Unexpected asset layout; refusing an incomplete bundle")
    css = (ROOT / "style.css").read_text(encoding="utf-8")
    # Keep the classic-script order and lexical scope of the browser entry point.
    js = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in SCRIPTS)
    if "</script" in js.lower() or "</style" in css.lower():
        raise ValueError("Assets contain an unsafe inline closing tag")
    return html.replace(css_tag, f"<style>{css}</style>").replace(
        script_tags, f"<script>{js}</script>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New HTML filename (must not exist)")
    args = parser.parse_args()
    try:
        html = build()
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(html)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not build Tone Lab: {exc}\n")
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()

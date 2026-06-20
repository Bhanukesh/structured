#!/usr/bin/env python3
"""
Extract structured data from unstructured documents (HTML, Markdown, plain text)
using the Anthropic Claude API, conforming to a user-supplied JSON Schema.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python extract_structured_data.py --input doc.html --schema schema.json --output result.json

Dependencies:
    pip install anthropic markdownify jsonschema
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

HTML_EXTS = {".html", ".htm", ".xhtml"}
MARKDOWN_EXTS = {".md", ".markdown", ".mdown", ".mkd"}
TEXT_EXTS = {".txt", ".text", ""}


def detect_type(path: Path) -> str:
    """Detect document type from the file extension."""
    ext = path.suffix.lower()
    if ext in HTML_EXTS:
        return "html"
    if ext in MARKDOWN_EXTS:
        return "markdown"
    if ext in TEXT_EXTS:
        return "text"
    # Unknown extension: treat as plain text.
    return "text"


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, retaining as much content as possible."""
    try:
        from markdownify import markdownify as md
    except ImportError:
        sys.exit("HTML input requires: pip install markdownify")

    # heading_style=ATX produces "# Heading"; keep tables, links, lists, etc.
    return md(
        html,
        heading_style="ATX",
        bullets="-",
        strip=[],          # don't strip any tags
        escape_asterisks=False,
        escape_underscores=False,
    )


def load_document(path: Path) -> str:
    """Read the input file and normalize it to Markdown/plain text."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    doc_type = detect_type(path)
    if doc_type == "html":
        return html_to_markdown(raw)
    return raw  # markdown and text are passed through unchanged


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-8"


def build_prompt(document: str, schema: dict) -> str:
    return (
        "You are a precise data-extraction engine. Extract structured data from "
        "the document below according to the provided JSON Schema.\n\n"
        "Rules:\n"
        "- Return ONLY a single JSON object that validates against the schema.\n"
        "- No markdown fences, no commentary, no preamble.\n"
        "- Use null (or omit optional fields) when a value is not present in the document.\n"
        "- Do not invent values that are not supported by the document.\n\n"
        f"JSON Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Document:\n<document>\n{document}\n</document>"
    )


def extract(document: str, schema: dict, client: anthropic.Anthropic) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system="You output only valid JSON conforming to the requested schema.",
        messages=[{"role": "user", "content": build_prompt(document, schema)}],
    )

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Defensively strip code fences if the model added them.
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\n---\n{text}")


def validate(data: dict, schema: dict) -> None:
    """Validate the extracted data against the schema, if jsonschema is available."""
    try:
        import jsonschema
    except ImportError:
        print("Note: install jsonschema to validate output.", file=sys.stderr)
        return
    jsonschema.validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured data from a document using Claude."
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Path to the input document (.html/.md/.txt).")
    parser.add_argument("--schema", "-s", required=True, type=Path,
                        help="Path to the JSON Schema describing the output.")
    parser.add_argument("--output", "-o", type=Path,
                        help="Where to write the JSON result (default: stdout).")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip JSON Schema validation of the result.")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Input file not found: {args.input}")
    if not args.schema.exists():
        sys.exit(f"Schema file not found: {args.schema}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set the ANTHROPIC_API_KEY environment variable.")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    document = load_document(args.input)

    client = anthropic.Anthropic()
    data = extract(document, schema, client)

    if not args.no_validate:
        try:
            validate(data, schema)
        except Exception as e:
            print(f"Warning: output failed schema validation: {e}", file=sys.stderr)

    out = json.dumps(data, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()

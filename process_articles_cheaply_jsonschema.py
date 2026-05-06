"""
process_articles_cheaply_jsonschema.py

Cheap/resumable Medium-export metadata builder.

What it does:
- Iterates over Medium HTML export files.
- Skips articles already present in the output JSON, unless source file changed and
  --reprocess-changed is used.
- Makes ONE OpenAI API call per article, returning structured JSON with:
  teaser, summary, keywords, video relevance score, video ideas, and notes.
- Validates the returned metadata with jsonschema before saving.
- Saves after each successful article so the job can be stopped/restarted safely.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from markdownify import markdownify
from openai import OpenAI
# Note: We're not manually constructing HTTP requests with requests.post() or headers herein.
# The OpenAI Python package handles that stuff! Our code just calls a method, and the library 
# handles the network request, authentication headers, serialization, & response parsing. Kthx.

# CONFIGS
DEFAULT_POSTS_DIR = r"D:\Medium\medium-export-20260501\posts"
DEFAULT_OUTPUT_FILE = "articles-metadata.json"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_CHARS = 10000
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_START_AFTER_FILENAME = ""


# This schema is sent to OpenAI. It is intentionally strict because OpenAI's
# structured outputs work best when the requested output shape is exact.
METADATA_SCHEMA: Dict[str, Any] = {
    "name": "article_metadata",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "teaser": {"type": "string"},
            "summary": {"type": "string"},
            "keywords": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "broad_topics": {"type": "array", "items": {"type": "string"}},
                    "specific_long_tail": {"type": "array", "items": {"type": "string"}},
                    "conceptual_thematic": {"type": "array", "items": {"type": "string"}},
                    "tonal_style": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "broad_topics",
                    "specific_long_tail",
                    "conceptual_thematic",
                    "tonal_style",
                ],
            },
            "video": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                    "recommended_format": {"type": "string"},
                    "recommended_length": {"type": "string"},
                    "title_ideas": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "score",
                    "reason",
                    "recommended_format",
                    "recommended_length",
                    "title_ideas",
                ],
            },
            "content_type": {"type": "string"},
            "search_notes": {"type": "string"},
        },
        "required": [
            "teaser",
            "summary",
            "keywords",
            "video",
            "content_type",
            "search_notes",
        ],
    },
}


def make_permissive_validation_schema(openai_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a local validation schema from the OpenAI response schema.

    Difference from the OpenAI schema:
    - Required fields are still required.
    - Basic types are still checked.
    - Extra/bonus fields are allowed at every object level.

    This matches the practical goal: reject missing/broken metadata, but do not
    reject a useful response just because it includes an extra property.
    """
    schema = deepcopy(openai_schema["schema"])

    def allow_extra_properties(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node["additionalProperties"] = True
            for child in node.get("properties", {}).values():
                allow_extra_properties(child)
        elif node.get("type") == "array":
            allow_extra_properties(node.get("items"))

    allow_extra_properties(schema)
    return schema


VALIDATION_SCHEMA: Dict[str, Any] = make_permissive_validation_schema(METADATA_SCHEMA)
VALIDATOR = Draft202012Validator(VALIDATION_SCHEMA)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_existing_metadata(output_file: Path) -> Dict[str, Any]:
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(metadata: Dict[str, Any], output_file: Path) -> None:
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    tmp_file.replace(output_file)


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    return title or "Untitled"

def extract_medium_post_id(html: str) -> Optional[str]:
    match = re.search(r'https://medium\.com/p/([^"\'>\s?#]+)', html)
    return match.group(1) if match else None

def extract_header_image_url(html: str) -> Optional[str]:
    match = re.search(r'https://cdn-images-1\.medium\.com/[^"\'>\s]+', html)
    return match.group(0) if match else None

def extract_article_content(filepath: Path, max_chars: int) -> Optional[Dict[str, Any]]:
    try:
        html = filepath.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title_tag = soup.find("h1") or soup.find("title")
        title = clean_title(title_tag.get_text(" ", strip=True) if title_tag else filepath.stem)

        raw_text = soup.get_text("\n", strip=True)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

        md = markdownify(str(soup), heading_style="ATX")
        md = re.sub(r"\n{4,}", "\n\n\n", md).strip()

        source_text = md if len(md) < len(raw_text) * 1.4 else raw_text
        truncated_text = source_text[:max_chars]

        medium_post_id = extract_medium_post_id(html)
        header_image_url = extract_header_image_url(html)

        stat = filepath.stat()
        return {
            "title": title,
            "text": truncated_text,
            "raw_text_length": len(raw_text),
            "markdown_length": len(md),
            "source_size_bytes": stat.st_size,
            "source_mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "truncated": len(source_text) > max_chars,
            "medium_post_id": medium_post_id,
            "medium_url": f"https://medium.com/p/{medium_post_id}" if medium_post_id else None,
            "header_image_url": header_image_url,            
        }
    except Exception as e:
        print(f"✗ Error reading {filepath.name}: {e}")
        return None


def already_processed(
    existing: Dict[str, Any],
    filepath: Path,
    reprocess_changed: bool,
) -> bool:
    current = existing.get(filepath.name)
    if not current:
        return False

    if not reprocess_changed:
        return True

    stat = filepath.stat()
    previous_size = current.get("source_size_bytes")
    previous_mtime = current.get("source_mtime")
    current_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()

    return previous_size == stat.st_size and previous_mtime == current_mtime

# SYSTEM & USER Prompts / Messages...
# The `system` message tells the model its role and rules.
# The `user` message includes the article title, article text, and the metadata instructions.
def build_messages(title: str, article_text: str) -> List[Dict[str, str]]:
    system = (
        "You create rich metadata for a writer's archive of Medium articles. "
        "Be accurate, specific, searchable, and useful for later retrieval. "
        "Do not invent facts not supported by the article text. "
        "Keep the teaser hooky but not clickbait. "
        "Video score must be 1-10, where 10 means the article strongly lends itself "
        "to a compelling video and 1 means it probably should remain text-only."
    )

    user = f"""
Article title: {title}

Article text:
---
{article_text}
---

Create metadata with these rules:
- teaser: 2-4 engaging sentences.
- summary: 120-180 words, useful for search and later rediscovery.
- keywords: 12-18 total across the four keyword buckets.
- video.score: integer from 1 to 10.
- video.reason: explain the score briefly.
- video.recommended_format: examples: talking-head essay, narrated short, tutorial, slideshow essay, interview prompt, not recommended.
- video.recommended_length: practical length range, such as "3-5 minutes" or "not recommended".
- video.title_ideas: 2-3 plausible titles.
- content_type: brief classification, such as essay, tutorial, memoir, product commentary, satire, travel, fiction, etc.
- search_notes: one concise sentence about why someone might want to find this article later.
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def get_metadata_from_openai(
    client: OpenAI,
    model: str,
    title: str,
    article_text: str,
    temperature: float,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=build_messages(title, article_text),
        response_format={
            "type": "json_schema",
            "json_schema": METADATA_SCHEMA,
        },
    )

    # Here is the OpenAI response:
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty response.")

    return json.loads(content)


def normalize_video_score(metadata: Dict[str, Any]) -> None:
    """
    Normalize a present video.score to an int from 1-10.

    This is deliberately not allowed to create a missing video object or a missing
    score. Missing required fields should fail validation and stay unsaved.
    """
    video = metadata.get("video")
    if not isinstance(video, dict) or "score" not in video:
        return

    try:
        score = int(video["score"])
    except Exception:
        return

    video["score"] = max(1, min(10, score))


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Return True only when metadata has the required structure.

    Extra properties are allowed. Missing required properties and clearly wrong
    types are rejected.
    """
    errors = sorted(VALIDATOR.iter_errors(metadata), key=lambda e: list(e.path))
    if not errors:
        return True

    print("⚠️ Metadata failed jsonschema validation:")
    for error in errors:
        path = ".".join(str(part) for part in error.path) or "<root>"
        print(f"  - {path}: {error.message}")
    return False


def process_articles(args: argparse.Namespace) -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file or environment.")

    posts_dir = Path(args.posts_dir)
    output_file = Path(args.output_file)

    if not posts_dir.exists():
        raise FileNotFoundError(f"Posts directory not found: {posts_dir}")

    client = OpenAI()
    existing = load_existing_metadata(output_file)

    all_files = sorted(posts_dir.glob("*.html"))

    if args.start_after_filename:
        filenames = [f.name for f in all_files]

        if args.start_after_filename in filenames:
            start_index = filenames.index(args.start_after_filename) + 1
            all_files = all_files[start_index:]
            print(f"Starting after: {args.start_after_filename}")
        else:
            print(f"⚠️ start-after filename not found: {args.start_after_filename}")
            print("Continuing from beginning.")    

    candidates = [
        f for f in all_files
        if not f.name.lower().startswith("draft")
        and not already_processed(existing, f, args.reprocess_changed)
    ]

    batch = candidates[: args.batch_size]

    print(f"Existing metadata records: {len(existing)}")
    print(f"HTML files found: {len(all_files)}")
    print(f"Articles needing processing: {len(candidates)}")
    print(f"Processing this run: {len(batch)}")
    print(f"Model: {args.model}")
    print(f"Max chars/article: {args.max_chars}")

    if args.dry_run:
        print("\nDry run only. Files that would be processed:")
        for f in batch:
            print(f"- {f.name}")
        return

    for index, filepath in enumerate(batch, start=1):
        print(f"\n[{index}/{len(batch)}] Processing: {filepath.name}")

        article = extract_article_content(filepath, max_chars=args.max_chars)
        if not article:
            continue

        header_image_url = article.get("header_image_url")
        if header_image_url is None:
            print(f"↷ Skipping (no header image): {filepath.name}")
            continue

        try:
            ai_metadata = get_metadata_from_openai(
                client=client,
                model=args.model,
                title=article["title"],
                article_text=article["text"],
                temperature=args.temperature,
            )

            normalize_video_score(ai_metadata)

            if not validate_metadata(ai_metadata):
                print("✗ Metadata not saved. This article will be retried on the next run.")
                continue

            record = {
                "filename": filepath.name,
                "title": article["title"],
                "processed_at": utc_now_iso(),
                "model": args.model,
                "max_chars": args.max_chars,
                "truncated": article["truncated"],
                "raw_text_length": article["raw_text_length"],
                "markdown_length": article["markdown_length"],
                "source_size_bytes": article["source_size_bytes"],
                "source_mtime": article["source_mtime"],
                "medium_post_id": article["medium_post_id"],
                "medium_url": article["medium_url"],
                "header_image_url": article["header_image_url"],
                "metadata": ai_metadata,
            }

            existing[filepath.name] = record
            save_metadata(existing, output_file)

            score = ai_metadata.get("video", {}).get("score", "?")
            print(f"✓ Saved. Video score: {score}/10")

        except Exception as e:
            print(f"✗ Failed: {e}")
            print("  Saved progress up to previous successful article.")

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print("\nRun complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cheaply process Medium HTML exports into rich article metadata."
    )
    parser.add_argument("--posts-dir", default=DEFAULT_POSTS_DIR)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    # "Temperature" controls the model's creativity. 0.3 seems like a mid-level, but you can experiment
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument(
        "--reprocess-changed",
        action="store_true",
        help="Reprocess files if size/mtime changed since the saved metadata record.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would be processed without calling the API.",
    )
    parser.add_argument("--start-after-filename", default=DEFAULT_START_AFTER_FILENAME)
    return parser.parse_args()


if __name__ == "__main__":
    process_articles(parse_args())

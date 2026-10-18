"""Batch fact verification.

For every claim: Google Custom Search -> download and summarise the top articles -> ask Gemini for a
True/False verdict grounded only in those summaries. Progress is checkpointed, so an interrupted run
resumes where it stopped.

    python batch_fact_verifier.py --start 0 --end 50
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_INPUT = "data/politifact_factcheck_data.json"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MIN_SUMMARY_CHARS = 200
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

SYSTEM_PROMPT = """You are an automated fact-verification assistant. You will be given a structured input that contains:

- A factual claim at the top
- A list of summarized news articles that are relevant to that claim.

You must return:

**Verdict:** True or False
**Collective Summary:** Overall what the sources say
**Reasoning:** 2-4 sentence justification
**Sources Summary:** Bullet list of titles with one-line context

Only use the provided summaries. Be objective. Format your output in Markdown."""


# ---------------------------------------------------------------- input / output files

def load_statements(path, start, end):
    """Claims from a JSON-lines file, as (line_index, statement) for start <= index < end."""
    statements = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start and line.strip():
                statements.append((i, json.loads(line)["statement"]))
    return statements


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def write_json(path, data):
    """Write atomically so an interrupted run never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def merge_unique(existing, new, key="fact"):
    seen, merged = set(), []
    for item in list(existing) + list(new):
        k = item.get(key)
        if k and k not in seen:
            merged.append(item)
            seen.add(k)
    return merged


# ---------------------------------------------------------------- retrieval

def ensure_nltk_data():
    """newspaper3k's summariser needs NLTK's punkt tokenizer; without it every summary fails."""
    import nltk

    nltk.download("punkt_tab", quiet=True)


def google_search(query, api_key, cse_id, num=10):
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)

    def run(q):
        return service.cse().list(q=q, cx=cse_id, num=min(num, 10)).execute().get("items", [])

    try:
        return run(query)
    except HttpError as e:
        # The API rejects some long or quote-heavy claims with HTTP 400: retry once with a plain query.
        if getattr(e.resp, "status", None) == 400:
            plain = re.sub(r"[\"“”‘’]", "", query)[:200].strip()
            if plain and plain != query:
                return run(plain)
        raise


def summarize_article(url, timeout=10):
    """Download a page and return a short extractive summary, or None if it cannot be summarised."""
    import requests
    from newspaper import Article

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        response.raise_for_status()
        article = Article(url)
        article.set_html(response.text)
        article.parse()
        article.nlp()
        return article.summary or None
    except Exception:
        return None


def filter_top_articles(articles, top_k=6, min_chars=MIN_SUMMARY_CHARS):
    """Keep articles with a substantial summary, longest first."""
    kept = [a for a in articles if len((a.get("summary") or "").strip()) >= min_chars]
    kept.sort(key=lambda a: len(a["summary"].strip()), reverse=True)
    return kept[:top_k]


# ---------------------------------------------------------------- verification

def build_prompt(fact, articles):
    lines = [f"**Factual Claim:**\n{fact}\n\n**Relevant Articles:**\n"]
    for i, article in enumerate(articles, 1):
        lines.append(
            f"\nArticle {i}:\n"
            f"- **Title:** {article['title']}\n"
            f"- **URL:** {article['url']}\n"
            f"- **Snippet:** {article['snippet']}\n"
            f"- **Summary:** {article['summary']}\n"
        )
    return "".join(lines)


def extract_section(header, text):
    match = re.search(rf"\*\*{re.escape(header)}:\*\*\s*(.*?)(?=\n\*\*|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_bullet_list(header, text):
    section = extract_section(header, text)
    return re.findall(r"[*\-]\s+(.*)", section) if section else []


def parse_verification(fact, text):
    return {
        "fact": fact,
        "verdict": extract_section("Verdict", text),
        "collective_summary": extract_section("Collective Summary", text),
        "reasoning": extract_section("Reasoning", text),
        "sources_summary": extract_bullet_list("Sources Summary", text),
    }


def run_gemini_verification(prompt_text, api_key, model=DEFAULT_MODEL, retries=3):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0, system_instruction=SYSTEM_PROMPT)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt_text, config=config)
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text.strip()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))  # back off on rate limits and transient errors


def process_fact(fact, keys, model=DEFAULT_MODEL, delay=0.5, top_k=6):
    """Verify one claim. Returns (fact_result, full_output, parsed). Raises if no usable evidence."""
    results = google_search(fact, keys["google_api_key"], keys["cse_id"], num=10)
    enriched = []
    for result in results:
        url = result.get("link")
        summary = summarize_article(url) if url else None
        if summary:
            enriched.append(
                {"title": result.get("title"), "url": url, "snippet": result.get("snippet"), "summary": summary}
            )
        time.sleep(delay)

    top = filter_top_articles(enriched, top_k=top_k)
    if not top:
        raise ValueError("No valid articles found.")

    output = run_gemini_verification(build_prompt(fact, top), keys["gemini_api_key"], model=model)
    return (
        {"fact": fact, "articles": top},
        {"fact": fact, "gemini_output": output},
        parse_verification(fact, output),
    )


def verify_claim(fact, keys, model=DEFAULT_MODEL, delay=0.5, top_k=6):
    """Verify a single claim. Returns the parsed verdict plus the articles it was based on.

    Raises ValueError if no usable evidence can be found."""
    fact_result, _, parsed = process_fact(fact, keys, model, delay, top_k)
    return {**parsed, "articles": [{"title": a["title"], "url": a["url"]} for a in fact_result["articles"]]}


# ---------------------------------------------------------------- batch driver

def log_error(path, known, index, fact, error):
    line = f"[{index}] {fact[:100]}... -> {error}\n"
    if line not in known:
        known.add(line)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def run_verification_batch(
    keys,
    start_idx=0,
    end_idx=10,
    max_workers=4,
    input_file=DEFAULT_INPUT,
    output_dir=DEFAULT_OUTPUT_DIR,
    model=DEFAULT_MODEL,
    delay=0.5,
    top_k=6,
    save_every=5,
):
    """Verify claims start_idx..end_idx-1. Safe to re-run: finished claims are skipped.

    Workers only fetch and verify; this thread is the single writer of every output file."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "facts": os.path.join(output_dir, "fact_results.json"),
        "full": os.path.join(output_dir, "full_output.json"),
        "parsed": os.path.join(output_dir, "parsed_output.json"),
        "checkpoint": os.path.join(output_dir, "checkpoint.json"),
        "errors": os.path.join(output_dir, "error_log.txt"),
    }
    done = set(load_json(paths["checkpoint"], []))
    facts_out = load_json(paths["facts"], [])
    full_out = load_json(paths["full"], [])
    parsed_out = load_json(paths["parsed"], [])
    known_errors = set()
    if os.path.exists(paths["errors"]):
        with open(paths["errors"], "r", encoding="utf-8") as f:
            known_errors = set(f.readlines())

    todo = [(i, fact) for i, fact in load_statements(input_file, start_idx, end_idx) if i not in done]
    already_done = len(done)
    failed = 0
    print(f"{len(todo)} claims to verify with {max_workers} workers ({already_done} already done).")

    def flush():
        write_json(paths["facts"], merge_unique([], facts_out))
        write_json(paths["full"], merge_unique([], full_out))
        write_json(paths["parsed"], merge_unique([], parsed_out))
        write_json(paths["checkpoint"], sorted(done))

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_fact, fact, keys, model, delay, top_k): (index, fact) for index, fact in todo
            }
            for n, future in enumerate(tqdm(as_completed(futures), total=len(futures)), 1):
                index, fact = futures[future]
                try:
                    fact_result, full_output, parsed = future.result()
                except Exception as e:
                    failed += 1
                    log_error(paths["errors"], known_errors, index, fact, e)
                    continue
                facts_out.append(fact_result)
                full_out.append(full_output)
                parsed_out.append(parsed)
                done.add(index)
                if n % save_every == 0:
                    flush()
    finally:
        flush()  # also runs on Ctrl+C, so finished work is never lost

    newly_done = len(done) - already_done
    print(f"Done: {newly_done} verified, {failed} failed (see {paths['errors']}).")
    return {"verified": newly_done, "failed": failed}


def require_keys():
    names = {"gemini_api_key": "GEMINI_API_KEY", "google_api_key": "GOOGLE_API_KEY", "cse_id": "GOOGLE_CSE_ID"}
    keys = {k: os.getenv(env) for k, env in names.items()}
    missing = [names[k] for k, v in keys.items() if not v]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}. Copy .env.example to .env and fill it in.")
    return keys


def main():
    parser = argparse.ArgumentParser(description="Verify claims with Google Custom Search + Gemini.")
    parser.add_argument("--claim", help="verify this one claim, print the result as JSON and exit")
    parser.add_argument("--start", type=int, default=0, help="first claim index (inclusive)")
    parser.add_argument("--end", type=int, default=10, help="last claim index (exclusive)")
    parser.add_argument("--workers", type=int, default=4, help="parallel claims")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model (or set GEMINI_MODEL)")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSON-lines file of claims")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=0.5, help="seconds to wait between article downloads")
    args = parser.parse_args()

    keys = require_keys()
    ensure_nltk_data()
    if args.claim:
        try:
            result = verify_claim(args.claim, keys, model=args.model, delay=args.delay)
        except ValueError as e:
            sys.exit(f"Could not verify the claim: {e}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    run_verification_batch(
        keys,
        start_idx=args.start,
        end_idx=args.end,
        max_workers=args.workers,
        input_file=args.input,
        output_dir=args.output_dir,
        model=args.model,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

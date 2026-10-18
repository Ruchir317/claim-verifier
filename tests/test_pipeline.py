import json

import pytest

import batch_fact_verifier as bv

GEMINI_REPLY = """**Verdict:** False
**Collective Summary:** The sources say the claim is not supported.
**Reasoning:** Two outlets checked the numbers. Neither found them.
**Sources Summary:**
* Example News - reports the figure was wrong
- Fact Desk: no evidence for the claim
"""

KEYS = {"gemini_api_key": "g", "google_api_key": "k", "cse_id": "c"}


def test_parse_verification_extracts_every_field():
    parsed = bv.parse_verification("a claim", GEMINI_REPLY)
    assert parsed["verdict"] == "False"
    assert parsed["collective_summary"] == "The sources say the claim is not supported."
    assert parsed["reasoning"].startswith("Two outlets")
    assert parsed["sources_summary"] == [
        "Example News - reports the figure was wrong",
        "Fact Desk: no evidence for the claim",
    ]


def test_parse_verification_tolerates_missing_sections():
    parsed = bv.parse_verification("a claim", "no structure at all")
    assert parsed["verdict"] is None
    assert parsed["sources_summary"] == []


def test_filter_top_articles_drops_short_and_sorts_longest_first():
    articles = [
        {"title": "short", "summary": "x" * 50},
        {"title": "medium", "summary": "y" * 300},
        {"title": "long", "summary": "z" * 900},
        {"title": "none", "summary": None},
    ]
    kept = bv.filter_top_articles(articles, top_k=5)
    assert [a["title"] for a in kept] == ["long", "medium"]
    assert len(bv.filter_top_articles(articles, top_k=1)) == 1


def test_load_statements_respects_the_range(tmp_path):
    path = tmp_path / "claims.json"
    path.write_text("\n".join(json.dumps({"statement": f"claim {i}"}) for i in range(6)), encoding="utf-8")
    assert bv.load_statements(str(path), 2, 4) == [(2, "claim 2"), (3, "claim 3")]
    assert bv.load_statements(str(path), 5, 100) == [(5, "claim 5")]


def test_merge_unique_keeps_first_and_ignores_keyless_items():
    merged = bv.merge_unique([{"fact": "a", "v": 1}], [{"fact": "a", "v": 2}, {"fact": "b"}, {"v": 3}])
    assert merged == [{"fact": "a", "v": 1}, {"fact": "b"}]


def test_build_prompt_lists_each_article():
    prompt = bv.build_prompt("the claim", [{"title": "T", "url": "u", "snippet": "s", "summary": "S"}])
    assert "the claim" in prompt and "Article 1" in prompt and "**Title:** T" in prompt


@pytest.fixture
def fake_world(monkeypatch):
    """Replace the network: claim 'bad' has no usable articles, everything else gets two."""
    calls = {"gemini": 0}

    def fake_search(query, api_key, cse_id, num=10):
        if "bad" in query:
            return []
        return [{"link": "https://a.example", "title": "A", "snippet": "sa"}, {"link": "https://b.example", "title": "B", "snippet": "sb"}]

    monkeypatch.setattr(bv, "google_search", fake_search)
    monkeypatch.setattr(bv, "summarize_article", lambda url, timeout=10: "s" * 400)

    def fake_gemini(prompt, api_key, model=bv.DEFAULT_MODEL, retries=3):
        calls["gemini"] += 1
        return GEMINI_REPLY

    monkeypatch.setattr(bv, "run_gemini_verification", fake_gemini)
    return calls


def write_claims(tmp_path, statements):
    path = tmp_path / "claims.json"
    path.write_text("\n".join(json.dumps({"statement": s}) for s in statements), encoding="utf-8")
    return str(path)


def test_batch_writes_outputs_logs_failures_and_resumes(tmp_path, fake_world):
    claims = write_claims(tmp_path, ["good one", "bad one", "good two"])
    out = str(tmp_path / "out")
    run = dict(input_file=claims, output_dir=out, max_workers=2, delay=0, save_every=1)

    summary = bv.run_verification_batch(KEYS, 0, 3, **run)
    assert summary == {"verified": 2, "failed": 1}
    assert fake_world["gemini"] == 2

    parsed = json.load(open(f"{out}/parsed_output.json", encoding="utf-8"))
    assert sorted(p["fact"] for p in parsed) == ["good one", "good two"]
    assert all(p["verdict"] == "False" for p in parsed)
    assert sorted(json.load(open(f"{out}/checkpoint.json"))) == [0, 2]
    assert "No valid articles found." in open(f"{out}/error_log.txt", encoding="utf-8").read()

    # Re-running does no new Gemini work for finished claims; the failed one is retried.
    again = bv.run_verification_batch(KEYS, 0, 3, **run)
    assert again == {"verified": 0, "failed": 1}
    assert fake_world["gemini"] == 2
    assert open(f"{out}/error_log.txt", encoding="utf-8").read().count("bad one") == 1


def test_batch_survives_an_unexpected_error(tmp_path, fake_world, monkeypatch):
    def boom(prompt, api_key, model=bv.DEFAULT_MODEL, retries=3):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(bv, "run_gemini_verification", boom)
    claims = write_claims(tmp_path, ["only claim"])
    out = str(tmp_path / "out")
    summary = bv.run_verification_batch(KEYS, 0, 1, input_file=claims, output_dir=out, delay=0)
    assert summary == {"verified": 0, "failed": 1}
    assert "quota exceeded" in open(f"{out}/error_log.txt", encoding="utf-8").read()


def test_require_keys_names_what_is_missing(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CSE_ID"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as exc:
        bv.require_keys()
    assert "GEMINI_API_KEY" in str(exc.value) and ".env.example" in str(exc.value)

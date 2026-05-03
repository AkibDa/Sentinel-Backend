import json
import time
import logging
import re
from google import genai
from google.genai import types
from fastapi import HTTPException, APIRouter

from app.config import MODEL, BLOCKED_DOMAINS, TRUSTED_DOMAINS, GEMINI_API_KEY
from app.schemas import FactCheckResponse, SourceResult, PipelineMetadata, FactCheckRequest

router = APIRouter()

SYSTEM_PROMPT = """You are a fact-checking pipeline that verifies news headlines against reputable sources.

Pipeline stages you must execute internally:
1. EXTRACT — identify key entities (people, places, dates, organisations, statistics) and the core claim.
2. RETRIEVE — use Google Search grounding to find current, reputable coverage. Prefer authoritative sources: AP, Reuters, BBC, NPR, Nature, WHO, CDC, NASA, PolitiFact, Snopes.
3. FILTER — mentally discard results from low-credibility or satirical domains. Only use high-tier journalism and peer-reviewed/government sources.
4. ANALYZE — for each reputable source found, determine whether it Supports, Refutes, is Neutral toward, or Partially supports the claim.
5. VERDICT — synthesise findings into a clear, fair verdict.

CCRITICAL: Your response must be ONLY a raw JSON object. No explanation, no markdown, no ```json fences, no prose before or after, no [cite] annotations. Start your response with { and end with }.

The JSON must follow this exact schema:
{
  "verdict": "True|False|Misleading|Partially True|Unverified|Disputed",
  "confidence": "High|Medium|Low",
  "entities": ["string"],
  "searchQuery": "the search terms used",
  "summary": "2-3 sentences, no citations or [cite] tags",
  "nuance": "one sentence or null",
  "sources": [
    {
      "title": "Article headline",
      "domain": "example.com",
      "stance": "Supports|Refutes|Neutral|Partial",
      "snippet": "one sentence, no [cite] tags"
    }
  ]
}

Rules:
- verdict is "Unverified" when no reputable sources cover the claim.
- verdict is "Disputed" when credible sources genuinely disagree.
- Be precise and politically neutral.
- Never fabricate sources.
- Do not include misinformation domains.
- OUTPUT ONLY THE JSON OBJECT. NOTHING ELSE.
"""

def _filter_sources(sources: list[dict]) -> tuple[list[dict], int]:
    filtered = []
    for s in sources:
        domain = s.get("domain", "").lower().lstrip("www.")
        if any(domain == bd or domain.endswith("." + bd) for bd in BLOCKED_DOMAINS):
            continue
        filtered.append(s)

    trusted_count = sum(
        1 for s in filtered
        if any(
            s.get("domain", "").lower().lstrip("www.") == td
            or s.get("domain", "").lower().lstrip("www.").endswith("." + td)
            for td in TRUSTED_DOMAINS
        )
    )
    return filtered, trusted_count

def _extract_json(raw: str) -> dict:
  raw = raw.replace("{{", "{").replace("}}", "}")

  raw = raw.replace("\u201c", '"').replace("\u201d", '"')
  raw = raw.replace("\u2018", "'").replace("\u2019", "'")

  raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
  raw = re.sub(r"\s*```$", "", raw.strip())

  start = raw.find("{")
  if start == -1:
    raise ValueError("No JSON object found in response")

  depth = 0
  in_string = False
  escape_next = False

  for i, ch in enumerate(raw[start:], start):
    if escape_next:
      escape_next = False
      continue
    if ch == "\\" and in_string:
      escape_next = True
      continue
    if ch == '"':
      in_string = not in_string
      continue
    if in_string:
      continue
    if ch == "{":
      depth += 1
    elif ch == "}":
      depth -= 1
      if depth == 0:
        candidate = raw[start:i + 1]
        return json.loads(candidate)

  raise ValueError("No complete JSON object found in response")


def run_pipeline(headline: str) -> FactCheckResponse:
    t0 = time.monotonic()

    client = genai.Client(api_key=GEMINI_API_KEY)

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f'Fact-check this headline: "{headline}"',
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=8192,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")

    raw_text = ""
    try:
      candidate = response.candidates[0]
      finish = candidate.finish_reason.name

      if finish == "MAX_TOKENS":
        raise HTTPException(status_code=502, detail="Model response was truncated. Try a shorter headline.")

      if finish not in ("STOP",):
        raise HTTPException(status_code=502, detail=f"Model stopped unexpectedly: {finish}")

      raw_text = response.text
      result = _extract_json(raw_text)

    except HTTPException:
      raise
    except Exception as exc:  # full output, no truncation
      raise HTTPException(status_code=502, detail=f"Model output was not valid JSON: {exc}")

    raw_sources = result.get("sources", [])
    filtered_sources, trusted_count = _filter_sources(raw_sources)
    latency_ms = int((time.monotonic() - t0) * 1000)

    return FactCheckResponse(
        verdict=result.get("verdict", "Unverified"),
        summary=result.get("summary", ""),
        nuance=result.get("nuance"),
        sources=[
            SourceResult(
                title=s.get("title", ""),
                domain=s.get("domain", ""),
                stance=s.get("stance", "Neutral"),
                snippet=s.get("snippet", ""),
            )
            for s in filtered_sources
        ],
        metadata=PipelineMetadata(
            search_query=result.get("searchQuery", ""),
            entities=result.get("entities", []),
            confidence=result.get("confidence", "Low"),
            sources_found=len(raw_sources),
            trusted_sources_used=trusted_count,
            latency_ms=latency_ms,
            model=MODEL,
        ),
    )

@router.post("/factcheck", response_model=FactCheckResponse)
def factcheck(req: FactCheckRequest):
  if not GEMINI_API_KEY:
    raise HTTPException(status_code=500, detail="API key not configured.")

  return run_pipeline(req.headline)

import json
import time
import logging
from google import genai
from google.genai import types
from fastapi import HTTPException, APIRouter

from app.config import MODEL, BLOCKED_DOMAINS, TRUSTED_DOMAINS, GEMINI_API_KEY
from app.schemas import FactCheckResponse, SourceResult, PipelineMetadata, FactCheckRequest

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_PROMPT = """You are a fact-checking pipeline that verifies news headlines against reputable sources.

Pipeline stages you must execute internally:
1. EXTRACT — identify key entities (people, places, dates, organisations, statistics) and the core claim.
2. RETRIEVE — use Google Search grounding to find current, reputable coverage. Prefer authoritative sources: AP, Reuters, BBC, NPR, Nature, WHO, CDC, NASA, PolitiFact, Snopes.
3. FILTER — mentally discard results from low-credibility or satirical domains. Only use high-tier journalism and peer-reviewed/government sources.
4. ANALYZE — for each reputable source found, determine whether it Supports, Refutes, is Neutral toward, or Partially supports the claim.
5. VERDICT — synthesise findings into a clear, fair verdict.

Respond ONLY with a valid JSON object matching the requested schema.
Rules:
- verdict is "Unverified" when no reputable sources cover the claim.
- verdict is "Disputed" when credible sources genuinely disagree with each other.
- Be precise and politically neutral.
- Never fabricate sources. Only include sources you actually retrieved via search.
- Do not include domains from known misinformation sites even if they appear in results.
"""

def _filter_sources(sources: list[dict]) -> tuple[list[dict], int]:
    filtered = []
    for s in sources:
        domain = s.get("domain", "").lower().lstrip("www.")
        if any(domain == bd or domain.endswith("." + bd) for bd in BLOCKED_DOMAINS):
            logger.warning("Blocked domain in model output: %s", domain)
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
        max_output_tokens=1500,
        response_mime_type="application/json",
        tools=[types.Tool(google_search=types.GoogleSearch())],
      ),
    )
  except Exception as exc:
    logger.error("Gemini API error: %s", exc)
    raise HTTPException(status_code=502, detail=f"Upstream API error: {exc}")

  try:
    raw_text = response.text
    result = json.loads(raw_text)
  except (json.JSONDecodeError, ValueError, Exception) as exc:
    logger.error("JSON parse error or empty content. Raw output: %s", str(exc))
    raise HTTPException(status_code=502, detail="Model output was not valid JSON.")

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

@router.post("/factcheck", response_model=FactCheckResponse, tags=["pipeline"])
def factcheck(req: FactCheckRequest):
  if not GEMINI_API_KEY:
    raise HTTPException(status_code=500, detail="API key not configured.")

  logger.info("Fact-check request: %r", req.headline[:80])
  return run_pipeline(req.headline)

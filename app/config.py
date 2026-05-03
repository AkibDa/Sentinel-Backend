import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

MODEL = "gemini-2.5-flash"
MAX_HEADLINE_LENGTH = 500

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "ft.com", "bloomberg.com", "economist.com", "nature.com", "science.org",
    "who.int", "cdc.gov", "nih.gov", "nasa.gov", "un.org",
    "politifact.com", "snopes.com", "factcheck.org", "fullfact.org",
    "abcnews.go.com", "cbsnews.com", "nbcnews.com", "pbs.org",
    "time.com", "theatlantic.com", "newscientist.com", "scientificamerican.com",
}

BLOCKED_DOMAINS = {
    "naturalnews.com", "infowars.com", "breitbart.com", "thedailybuzz.com",
    "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net",
    "theonion.com", "clickhole.com",
}

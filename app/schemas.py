from pydantic import BaseModel, EmailStr, Field
from typing import Literal

class UserCreate(BaseModel):
  email: EmailStr
  password: str

class UserLogin(BaseModel):
  email: EmailStr
  password: str

class AnalyseRequest(BaseModel):
  type: Literal["image","video","audio","url"]
  input: str

class FactCheckRequest(BaseModel):
  headline: str = Field(..., min_length=5, max_length=500)
  language: str = Field(default="en", description="BCP-47 language tag")

class SourceResult(BaseModel):
  title: str
  domain: str
  stance: Literal["Supports", "Refutes", "Neutral", "Partial"]
  snippet: str

class PipelineMetadata(BaseModel):
  search_query: str
  entities: list[str]
  confidence: Literal["High", "Medium", "Low"]
  sources_found: int
  trusted_sources_used: int
  latency_ms: int
  model: str

class FactCheckResponse(BaseModel):
  verdict: Literal["True", "False", "Misleading", "Partially True", "Unverified", "Disputed"]
  summary: str
  nuance: str | None
  sources: list[SourceResult]
  metadata: PipelineMetadata

class HealthResponse(BaseModel):
  status: str
  model: str

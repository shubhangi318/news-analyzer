from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


# Request/response models
class CompanyRequest(BaseModel):
    company_name: str
    num_articles: Optional[int] = Field(default=10, ge=1, le=50)
    force_refresh: Optional[bool] = False
    articles: Optional[List[Dict[str, Any]]] = None
    keywords: Optional[List[str]] = None


class ArticleAnalysis(BaseModel):
    summary: str
    keywords: List[str]
    relevance: Literal["High", "Medium", "Low"]
    industry: Literal["Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Healthcare", "Financials", "Information Technology", "Communication Services", "Utilities", "Real Estate"]
    main_topic: Literal["Financial Results", "Mergers & Acquisitions", "Stock Market", "Corporate Strategy", "Layoffs & Restructuring", "Competitor & Market Trends", "Regulatory & Legal Issues", "Technology & Innovation", "Sustainability & ESG", "Executive Leadership", "Employee Culture", "Product Launch", "Crisis Management", "Government & Geopolitics"]
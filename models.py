from pydantic import BaseModel
from typing import List

class PriceAnalysisRequest(BaseModel):
    product_name: str
    current_price: float
    historical_prices: List[float]

class PriceAnalysisResponse(BaseModel):
    ml_trust_score: float
    category: str
    z_score: float
    reason: str

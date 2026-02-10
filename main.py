from fastapi import FastAPI
from models import PriceAnalysisRequest, PriceAnalysisResponse
from ml_logic import analyze_price

app = FastAPI(title="Price Trust ML Service")


@app.post("/analyze", response_model=PriceAnalysisResponse)
def analyze_price_api(request: PriceAnalysisRequest):
    return analyze_price(
        request.current_price,
        request.historical_prices
    )

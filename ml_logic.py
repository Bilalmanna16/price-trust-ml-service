import numpy as np

def analyze_price(current_price, historical_prices):
    # Not enough data
    if len(historical_prices) < 3:
        return {
            "ml_trust_score": 70.0,
            "category": "Insufficient Data",
            "z_score": 0.0,
            "reason": "Not enough historical prices for analysis"
        }

    mean = np.mean(historical_prices)
    std = np.std(historical_prices)

    if std == 0:
        return {
            "ml_trust_score": 70.0,
            "category": "Stable Price",
            "z_score": 0.0,
            "reason": "Prices show no variation"
        }

    z_score = abs((current_price - mean) / std)

    if z_score < 1:
        return {
            "ml_trust_score": 90.0,
            "category": "Fair",
            "z_score": z_score,
            "reason": "Price is within normal range"
        }
    elif z_score < 2:
        return {
            "ml_trust_score": 60.0,
            "category": "Slightly High",
            "z_score": z_score,
            "reason": "Price deviates moderately from average"
        }
    else:
        return {
            "ml_trust_score": 30.0,
            "category": "Suspicious",
            "z_score": z_score,
            "reason": "Price deviates significantly from historical pattern"
        }

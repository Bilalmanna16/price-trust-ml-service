import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


def build_features(prices):
    prices = np.array(prices)
    mean = prices.mean()
    std = prices.std() if prices.std() != 0 else 1

    features = []
    for p in prices:
        z = (p - mean) / std
        features.append([p, z])

    return np.array(features), mean, std


def kmeans_trust_score(features, current_price):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(features)

    centers = kmeans.cluster_centers_
    distances = [abs(current_price - c[0]) for c in centers]

    min_dist = min(distances)
    score = 100 - (min_dist / max(current_price, 1)) * 100

    return max(0, min(100, score))


def isolation_forest_score(features, current_feature):
    model = IsolationForest(
        contamination=0.2,
        random_state=42
    )
    model.fit(features)

    raw_score = model.decision_function([current_feature])[0]
    normalized = (raw_score + 0.5) * 100

    return max(0, min(100, normalized))


def analyze_price(current_price, historical_prices):

    # Safety check
    if len(historical_prices) < 3:
        return {
            "ml_trust_score": 50.0,
            "category": "Insufficient Data",
            "kmeans_score": 0.0,
            "isolation_score": 0.0,
            "reason": "Not enough historical prices for ML analysis"
        }

    # Feature engineering
    features, mean, std = build_features(historical_prices)

    current_z = (current_price - mean) / std
    current_feature = [current_price, current_z]

    # ML scores
    kmeans_score = kmeans_trust_score(features, current_price)
    isolation_score = isolation_forest_score(features, current_feature)

    # Final combined score
    final_score = round((0.5 * kmeans_score) + (0.5 * isolation_score), 2)

    # Category
    if final_score >= 75:
        category = "Fair"
    elif final_score >= 50:
        category = "Moderate"
    else:
        category = "Suspicious"

    return {
        "ml_trust_score": final_score,
        "category": category,
        "kmeans_score": round(kmeans_score, 2),
        "isolation_score": round(isolation_score, 2),
        "reason": "Score derived using K-Means clustering and Isolation Forest"
    }

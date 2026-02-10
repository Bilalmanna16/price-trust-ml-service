import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# ---------------- LOAD BASELINE CSV ----------------

BASELINE_DATA = {}

with open("baseline_prices.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        BASELINE_DATA[row["category"].lower()] = {
            "min": float(row["min_price"]),
            "avg": float(row["avg_price"]),
            "max": float(row["max_price"])
        }


def normalize(text: str):
    return text.lower().strip()


# ---------------- BASELINE (COLD START) ----------------

def baseline_trust_score(product_name, price):
    key = normalize(product_name)

    if key not in BASELINE_DATA:
        return {
            "ml_trust_score": 50.0,
            "category": "Unknown",
            "kmeans_score": 0.0,
            "isolation_score": 0.0,
            "reason": "No baseline data available for this product category"
        }

    ref = BASELINE_DATA[key]

    if price < ref["min"] * 0.75 or price > ref["max"] * 1.25:
        score = 30.0
        category = "Suspicious"
    elif ref["min"] <= price <= ref["max"]:
        score = 80.0
        category = "Fair"
    else:
        score = 55.0
        category = "Moderate"

    return {
        "ml_trust_score": score,
        "category": category,
        "kmeans_score": 0.0,
        "isolation_score": 0.0,
        "reason": "Cold-start decision using baseline market price dataset"
    }


# ---------------- PERSONALIZED ML ----------------

def build_features(prices):
    prices = np.array(prices)
    mean = prices.mean()
    std = prices.std() if prices.std() != 0 else 1

    features = []
    for p in prices:
        z = (p - mean) / std
        features.append([p, z])

    return np.array(features), mean, std


def kmeans_score(features, current_price):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(features)

    centers = kmeans.cluster_centers_
    distances = [abs(current_price - c[0]) for c in centers]
    min_dist = min(distances)

    score = 100 - (min_dist / max(current_price, 1)) * 100
    return max(0, min(100, score))


def isolation_score(features, current_feature):
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(features)

    raw = model.decision_function([current_feature])[0]
    normalized = (raw + 0.5) * 100
    return max(0, min(100, normalized))


# ---------------- MAIN ENTRY ----------------

def analyze_price(product_name, current_price, historical_prices):

    # ---- COLD START ----
    if len(historical_prices) < 3:
        return baseline_trust_score(product_name, current_price)

    # ---- PERSONALIZED ML ----
    features, mean, std = build_features(historical_prices)
    current_z = (current_price - mean) / std
    current_feature = [current_price, current_z]

    k_score = kmeans_score(features, current_price)
    i_score = isolation_score(features, current_feature)

    final_score = round((0.5 * k_score) + (0.5 * i_score), 2)

    if final_score >= 75:
        category = "Fair"
    elif final_score >= 50:
        category = "Moderate"
    else:
        category = "Suspicious"

    return {
        "ml_trust_score": final_score,
        "category": category,
        "kmeans_score": round(k_score, 2),
        "isolation_score": round(i_score, 2),
        "reason": "Personalized ML analysis using historical price patterns"
    }

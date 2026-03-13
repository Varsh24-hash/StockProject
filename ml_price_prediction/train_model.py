import pandas as pd
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report


# ===============================
# 1. Load all processed datasets
# ===============================

processed_folder = "data/processed"

files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

all_data = []

for file in files:

    filepath = os.path.join(processed_folder, file)

    df = pd.read_csv(filepath)

    # extract ticker name
    ticker = file.split("_")[0]

    df["ticker"] = ticker

    all_data.append(df)

# combine all stocks into one dataset
data = pd.concat(all_data, ignore_index=True)

print("Total dataset shape:", data.shape)


# ===============================
# 2. Select ML features
# ===============================

features = [
    "return_1d",
    "MA_10",
    "MA_50",
    "volatility",
    "volume_change",
    "RSI"
]

X = data[features]
y = data["target"]


# ===============================
# 3. Train / Test Split
# ===============================

split = int(len(data) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ===============================
# 4. Logistic Regression
# ===============================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\nLogistic Regression Accuracy:",
      accuracy_score(y_test, log_pred))

print(classification_report(y_test, log_pred))


# ===============================
# 5. Random Forest
# ===============================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

print(classification_report(y_test, rf_pred))


# ===============================
# 6. XGBoost
# ===============================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

print("\nXGBoost Accuracy:",
      accuracy_score(y_test, xgb_pred))

print(classification_report(y_test, xgb_pred))


# ===============================
# 7. Prediction Probabilities
# ===============================

probs = xgb_model.predict_proba(X_test)[:,1]

print("\nSample prediction probabilities:")
print(probs[:10])
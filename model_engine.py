import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_and_predict(live_metrics_dict):
    """
    Trains an XGBoost model using realistic market distributions and Graham-based fundamental heuristics.
    Predicts the health of a live stock and returns explainability matrices.
    """
    # ==========================================
    # 1. REALISTIC MARKET DATA GENERATION
    # ==========================================
    np.random.seed(42) # Ensures consistent results during interviews
    n_samples = 5000

    # Generate statistically realistic distributions for Indian/US equities
    roe = np.clip(np.random.normal(12, 15, n_samples), -50, 100)
    opm = np.clip(np.random.normal(15, 12, n_samples), -50, 100)
    # Debt is typically log-normally distributed (many low debt, few extremely high debt)
    debt_to_equity = np.clip(np.random.lognormal(mean=0.2, sigma=0.8, size=n_samples), 0, 10) 
    current_ratio = np.clip(np.random.normal(1.5, 0.8, n_samples), 0.1, 10)
    pe_ratio = np.clip(np.random.normal(25, 20, n_samples), 1, 200)

    df = pd.DataFrame({
        'Return on Equity (ROE)': roe,
        'Operating Margin (OPM)': opm,
        'Debt to Equity Ratio': debt_to_equity,
        'Current Ratio': current_ratio,
        'P/E Ratio': pe_ratio
    })

    # ==========================================
    # 2. THE BENJAMIN GRAHAM HEURISTIC LABELING
    # ==========================================
    # We label the "ground truth" based on strict, universally accepted fundamental rules
    health_score = np.zeros(n_samples)
    
    # Add points for strength
    health_score += np.where(df['Return on Equity (ROE)'] > 15, 1, 0)
    health_score += np.where(df['Operating Margin (OPM)'] > 12, 1, 0)
    health_score += np.where(df['Current Ratio'] > 1.2, 1, 0)
    
    # Subtract points for risk
    health_score -= np.where(df['Debt to Equity Ratio'] > 1.5, 1, 0)
    health_score -= np.where(df['P/E Ratio'] > 40, 1, 0)

    # Base target: If score > 0, it's a fundamentally sound company (1)
    target = np.where(health_score >= 1, 1, 0)

    # Inject 10% random noise (Because the real market is never 100% logical)
    noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    target[noise_indices] = 1 - target[noise_indices] # Flip 10% of the labels

    df['Target'] = target

    X = df.drop('Target', axis=1)
    y = df['Target']

    # ==========================================
    # 3. TRAIN THE XGBOOST ALGORITHM
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the gradient booster to reverse-engineer the fundamental rules
    model = xgb.XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=5, 
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # ==========================================
    # 4. LIVE STOCK PREDICTION
    # ==========================================
    live_df = pd.DataFrame([live_metrics_dict])
    live_df = live_df[X.columns]

    prediction = int(model.predict(live_df)[0])
    probability = float(model.predict_proba(live_df)[0][1])

    # ==========================================
    # 5. EXPLAINABLE AI (SHAP DECONSTRUCTION)
    # ==========================================
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(live_df)

    return prediction, probability, accuracy, explainer, shap_values, live_df
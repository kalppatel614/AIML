import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import shap

# 1. GENERATE SYNTHETIC DATA (10k rows)
def generate_data():
    np.random.seed(42)
    n = 10000
    data = {
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(20000, 150000, n),
        'loan_amount': np.random.randint(1000, 50000, n),
        'credit_score': np.random.randint(300, 850, n),
        'employment_years': np.random.randint(0, 40, n)
    }
    df = pd.DataFrame(data)
    # Define target logic: Higher debt/low score = high risk
    df['risk'] = (df['loan_amount'] / df['income'] > 0.4) | (df['credit_score'] < 500)
    df['risk'] = df['risk'].astype(int)
    return df

# 2. PREPROCESSING & TRAINING
df = generate_data()
X = df.drop('risk', axis=1)
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression (Baseline)
lr = LogisticRegression().fit(X_train_scaled, y_train)

# Model 2: Random Forest (Main Model)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train_scaled, y_train)

# 3. SAVE MODEL AND TOOLS
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 4. SHAP EXPLAINER
explainer = shap.TreeExplainer(rf)
with open('explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

print(f"Model Trained! RF AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]):.2f}")
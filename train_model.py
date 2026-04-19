# import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import roc_auc_score
# import shap

# # 1. GENERATE SYNTHETIC DATA (10k rows)
# def generate_data():
#     np.random.seed(42)
#     n = 10000
#     data = {
#         'person_age': np.random.randint(18, 70, n),
#         'person_income': np.random.randint(20000, 150000, n),
#         'loan_amnt': np.random.randint(1000, 50000, n),
#         'cb_person_cred_hist_length': np.random.randint(300, 850, n),
#         'person_emp_length': np.random.randint(0, 40, n)
#     }
#     df = pd.DataFrame(data)
#     # Define target logic: Higher debt/low score = high risk
#     df['risk'] = (df['loan_amnt'] / df['person_income'] > 0.4) | (df['cb_person_cred_hist_length'] < 500)
#     df['risk'] = df['risk'].astype(int)
#     return df

# # 2. PREPROCESSING & TRAINING
# df = generate_data()
# X = df.drop('risk', axis=1)
# y = df['risk']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Model 1: Logistic Regression (Baseline)
# lr = LogisticRegression().fit(X_train_scaled, y_train)

# # Model 2: Random Forest (Main Model)
# rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42).fit(X_train_scaled, y_train)

# # 3. SAVE MODEL AND TOOLS
# with open('credit_model.pkl', 'wb') as f:
#     pickle.dump(rf, f)
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# # 4. SHAP EXPLAINER
# explainer = shap.TreeExplainer(rf)
# with open('explainer.pkl', 'wb') as f:
#     pickle.dump(explainer, f)

# print(f"Model Trained! RF AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]):.2f}")




# import pandas as pd
# import numpy as np
# import pickle
# import glob
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler

# # --- DYNAMIC FILE LOADING ---
# # This finds any file ending in .csv in your current directory
# csv_files = glob.glob("*.csv")

# if not csv_files:
#     print("❌ Error: No CSV file found in the folder. Please add your dataset.")
# else:
#     # Pick the first CSV found
#     file_path = csv_files[0]
#     print(f"📂 Found dataset: {file_path}. Starting training...")
    
#     df = pd.read_csv(file_path)
    
#     # 2. PREPROCESSING
#     df = df.dropna()

#     # Define features (Ensure these exist in your CSV)
#     FEATURE_NAMES = [
#         'person_age', 
#         'person_income', 
#         'loan_amnt', 
#         'cb_person_cred_hist_length', 
#         'person_emp_length'
#     ]

#     # Verify if columns exist
#     if all(col in df.columns for col in FEATURE_NAMES):
#         X = df[FEATURE_NAMES]
#         # Using 'loan_status' as standard target in this dataset
#         y = df['loan_status'] 

#         # 3. SPLIT AND SCALE
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)

#         # 4. TRAIN MODEL
#         rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
#         rf.fit(X_train_scaled, y_train)

#         # 5. SAVE
#         with open('credit_model.pkl', 'wb') as f:
#             pickle.dump(rf, f)

#         with open('scaler.pkl', 'wb') as f:
#             pickle.dump(scaler, f)

#         print(f"✅ Success! Trained on '{file_path}' and saved model assets.")
#     else:
#         print(f"❌ Error: The file {file_path} is missing required columns: {FEATURE_NAMES}")




import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap

# 1. GENERATE SYNTHETIC DATA
def generate_synthetic_credit_data():
    np.random.seed(42)
    n = 10000
    
    # We use the EXACT column names expected by FEATURE_NAMES in app.py
    data = {
        'person_age': np.random.randint(18, 70, n),
        'person_income': np.random.randint(20000, 150000, n),
        'loan_amnt': np.random.randint(1000, 50000, n),
        'cb_person_cred_hist_length': np.random.randint(1, 30, n), # Years of history
        'person_emp_length': np.random.randint(0, 40, n)
    }
    
    df = pd.DataFrame(data)
    
    # Logic for synthetic risk (loan_status):
    # High risk if Loan-to-Income is high (> 40%) OR employment is very short
    dti = df['loan_amnt'] / df['person_income']
    df['loan_status'] = ((dti > 0.45) | (df['person_emp_length'] < 1)).astype(int)
    
    return df

# 2. TRAINING PROCESS
df = generate_synthetic_credit_data()

# Ensure order matches FEATURE_NAMES in app.py
FEATURE_NAMES = ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length', 'person_emp_length']

X = df[FEATURE_NAMES]
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# 3. SAVE ASSETS
# These filenames must match what is in your app.py load_assets() function
with open('credit_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model Trained on Synthetic Data!")
print(f"Features used: {FEATURE_NAMES}")
print("Saved files: credit_model.pkl, scaler.pkl")
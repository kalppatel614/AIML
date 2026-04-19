# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

# # --- CONFIG & ASSETS ---
# st.set_page_config(page_title="CreditGuard AI", layout="wide", initial_sidebar_state="collapsed")

# # Custom CSS for a clean look
# st.markdown("""
#     <style>
#     .main { background-color: #f8f9fa; }
#     .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
#     </style>
#     """, unsafe_allow_html=True)

# @st.cache_resource
# def load_assets():
#     model = pickle.load(open('model.pkl', 'rb'))
#     scaler = pickle.load(open('scaler.pkl', 'rb'))
#     explainer = pickle.load(open('explainer.pkl', 'rb'))
#     return model, scaler, explainer

# model, scaler, explainer = load_assets()

# # --- HEADER ---
# st.title("🛡️ CreditGuard AI")
# st.markdown("Professional Grade Credit Risk Assessment & Interpretability Portal")
# st.divider()

# tabs = st.tabs(["👤 Individual Assessment", "📊 Batch Processing"])

# # --- TAB 1: INDIVIDUAL ASSESSMENT ---
# with tabs[0]:
#     with st.container():
#         col1, col2 = st.columns([1, 2], gap="large")
        
#         with col1:
#             st.subheader("Customer Profile")
#             age = st.number_input("Age", 18, 100, 30)
#             income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
#             loan = st.number_input("Loan Amount Requested ($)", 0, 500000, 10000)
#             score = st.slider("Credit Score", 300, 850, 650)
#             emp_years = st.number_input("Years of Employment", 0, 50, 5)
            
#             predict_btn = st.button("Generate Risk Report", type="primary", use_container_width=True)

#         with col2:
#             if predict_btn:
#                 # Calculations
#                 features = np.array([[age, income, loan, score, emp_years]])
#                 features_scaled = scaler.transform(features)
#                 prob = model.predict_proba(features_scaled)[0][1]
                
#                 # Financial Metric: Debt-to-Income
#                 dti = (loan / income) if income > 0 else 0
                
#                 # 1. GAUGE CHART
#                 fig_gauge = go.Figure(go.Indicator(
#                     mode = "gauge+number",
#                     value = prob * 100,
#                     domain = {'x': [0, 1], 'y': [0, 1]},
#                     title = {'text': "Default Probability (%)", 'font': {'size': 24}},
#                     gauge = {
#                         'axis': {'range': [None, 100]},
#                         'bar': {'color': "black"},
#                         'steps': [
#                             {'range': [0, 30], 'color': "#2ecc71"},
#                             {'range': [30, 70], 'color': "#f1c40f"},
#                             {'range': [70, 100], 'color': "#e74c3c"}
#                         ],
#                     }
#                 ))
#                 fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
#                 st.plotly_chart(fig_gauge, use_container_width=True)

#                 # 2. ANALYSIS CARDS
#                 st.markdown("### Decision Summary")
#                 m1, m2, m3 = st.columns(3)
                
#                 with m1:
#                     if prob > 0.7:
#                         st.error(f"**Risk Status**\n\n HIGH ({prob:.1%})")
#                     elif prob > 0.3:
#                         st.warning(f"**Risk Status**\n\n MEDIUM ({prob:.1%})")
#                     else:
#                         st.success(f"**Risk Status**\n\n LOW ({prob:.1%})")

#                 with m2:
#                     # DTI usually risky if > 40%
#                     if dti > 0.4:
#                         st.error(f"**Loan-to-Income**\n\n {dti:.2%} (High)")
#                     else:
#                         st.success(f"**Loan-to-Income**\n\n {dti:.2%} (Healthy)")

#                 with m3:
#                     if score < 580:
#                         st.error(f"**Score Impact**\n\n Negative ({score})")
#                     elif score < 700:
#                         st.warning(f"**Score Impact**\n\n Neutral ({score})")
#                     else:
#                         st.success(f"**Score Impact**\n\n Positive ({score})")

#                 # 3. CLEAN SHAP EXPLANATION
#                 st.subheader("🔍 AI Decision Factors")
#                 st.info("The chart below shows which features pushed the risk score up (Red) or down (Blue).")
                
#                 shap_values = explainer(features_scaled)
#                 # Handle SHAP multi-output safely
#                 if isinstance(shap_values, list):
#                     sv = shap_values[1]
#                 elif len(shap_values.shape) == 3:
#                     sv = shap_values[0, :, 1]
#                 else:
#                     sv = shap_values[0]

#                 fig_shap, ax = plt.subplots(figsize=(8, 4))
#                 shap.plots.bar(sv, show=False)
#                 plt.yticks(range(5), ['Age', 'Income', 'Loan Amt', 'Credit Score', 'Experience'])
#                 st.pyplot(fig_shap)
#             else:
#                 st.info("← Fill in details and click 'Generate Risk Report' to see the analysis.")

# # --- TAB 2: BATCH PROCESSING ---
# with tabs[1]:
#     uploaded_file = st.file_uploader("Drop Credit Application File (CSV or XLSX)", type=["csv", "xlsx"])
    
#     if uploaded_file:
#         df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
#         scaled_batch = scaler.transform(df_input)
#         preds_prob = model.predict_proba(scaled_batch)[:, 1]
        
#         df_input['Risk_Score'] = preds_prob
#         df_input['Decision'] = ["❌ Decline" if p > 0.5 else "✅ Approve" for p in preds_prob]
        
#         st.subheader("Batch Results")
        
#         c1, c2 = st.columns([2, 1])
#         with c1:
#             st.dataframe(df_input.style.background_gradient(subset=['Risk_Score'], cmap='RdYlGn_r'), use_container_width=True)
#         with c2:
#             fig_pie = px.pie(df_input, names='Decision', color='Decision', 
#                              color_discrete_map={'✅ Approve':'#2ecc71','❌ Decline':'#e74c3c'})
#             st.plotly_chart(fig_pie, use_container_width=True)

#         st.download_button("Download Processed Report", df_input.to_csv(index=False), "Batch_Report.csv", "text/csv")




import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_shap import st_shap

# --- MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# Load model and scaler
@st.cache_resource
def load_assets():
    with open('credit_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

st.title("🛡️ Credit Risk Intelligence Dashboard")

tab1, tab2 = st.tabs(["Individual Assessment", "Batch Processing & Analytics"])

# Define feature names globally to keep order consistent
FEATURE_NAMES = ['person_age', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length', 'person_emp_length']

# --- TAB 1: INDIVIDUAL ASSESSMENT (UPDATED VISUALS) ---
with tab1:
    st.header("Individual Loan Risk Predictor")
   
    with st.expander("Input Features (Click to Edit)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 25)
            income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
            emp_length = st.number_input("Years of Employment", 0, 50, 2)
        with col2:
            loan_amt = st.number_input("Loan Amount ($)", 0, 500000, 10000)
            hist_len = st.number_input("Credit History Length (Years)", 0, 50, 5)

    if st.button("Predict Risk Score", type="primary"):
        # Convert input to DataFrame
        input_df = pd.DataFrame([[age, income, loan_amt, emp_length, hist_len]], columns=FEATURE_NAMES)
       
        # Scale the features
        features_scaled = scaler.transform(input_df)
       
        # Prediction
        prob = model.predict_proba(features_scaled)[0][1]
        risk_score = round(prob * 100, 2)
       
        # 1. NEW GAUGE VISUALIZATION (Plotly)
        st.write("---")
       
        # Calculate derived metrics for the Summary
        dti = round((loan_amt / income * 100), 1) if income > 0 else 0
       
        # Create Gauge Plot
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            title = {'text': "Default Probability (%)"},
            delta = {'reference': 50, 'relative': False}, # Shows difference from random chance
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "white", 'thickness': 0.15}, # Slim pointer bar
                'bgcolor': "rgba(0,0,0,0)", # Transparent background
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "green"}, # Low Risk
                    {'range': [30, 70], 'color': "gold"},  # Medium Risk
                    {'range': [70, 100], 'color': "crimson"} # High Risk
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
       
        # Define layout settings
        fig.update_layout(
            font = {'color': "white", 'family': "Arial"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=30, r=30, t=60, b=0),
            height=300
        )
       
        st.plotly_chart(fig, use_container_width=True)

        # 2. DECISION SUMMARY CARDS (Mimicking Screenshot)
        st.subheader("Decision Summary")
       
        col_s1, col_s2, col_s3 = st.columns(3)
       
        # Define status colors
        if risk_score < 30:
            status_color = "green"
            status_text = f"LOW ({risk_score}%)"
        elif risk_score < 70:
            status_color = "#D4AF37" # Gold
            status_text = f"MEDIUM ({risk_score}%)"
        else:
            status_color = "crimson"
            status_text = f"HIGH ({risk_score}%)"

        # Using markdown with style for the cards
        col_s1.markdown(f"""
            <div style='background-color: {status_color}22; padding: 20px; border-radius: 10px; border: 2px solid {status_color};'>
                <p style='color: {status_color}; font-weight: bold; margin: 0;'>Risk Status</p>
                <p style='color: {status_color}; font-size: 24px; font-weight: bold; margin: 0;'>{status_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # DTI Card
        if dti < 35: dti_color, dti_text = "green", f"{dti}% (Healthy)"
        elif dti < 50: dti_color, dti_text = "#D4AF37", f"{dti}% (Manual Review)"
        else: dti_color, dti_text = "crimson", f"{dti}% (Critical DTI)"

        col_s2.markdown(f"""
            <div style='background-color: {dti_color}22; padding: 20px; border-radius: 10px; border: 2px solid {dti_color};'>
                <p style='color: {dti_color}; font-weight: bold; margin: 0;'>Loan-to-Income</p>
                <p style='color: {dti_color}; font-size: 24px; font-weight: bold; margin: 0;'>{dti_text}</p>
            </div>
        """, unsafe_allow_html=True)

        # Credit History Card (Using the Hist Len input for now)
        if hist_len > 8: h_color, h_text = "green", f"Veteran ({hist_len} yrs)"
        elif hist_len > 3: h_color, h_text = "#D4AF37", f"Average ({hist_len} yrs)"
        else: h_color, h_text = "crimson", f"Thin File ({hist_len} yrs)"

        col_s3.markdown(f"""
            <div style='background-color: {h_color}22; padding: 20px; border-radius: 10px; border: 2px solid {h_color};'>
                <p style='color: {h_color}; font-weight: bold; margin: 0;'>Score Impact</p>
                <p style='color: {h_color}; font-size: 24px; font-weight: bold; margin: 0;'>{h_text}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("---")

        # --- UPDATED SHAP LOGIC FOR v0.20+ ---
        st.subheader("Why this score? (SHAP Explainability)")
       
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_scaled)

            # Extract base value for Class 1 (Default)
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value

            # Extract SHAP values for Class 1 (Default)
            if isinstance(shap_values, list):
                # Ensure it's 1D for the plot
                sv = shap_values[1][0]
            else:
                # In some versions, Random Forest returns 3D array [samples, features, classes]
                if len(shap_values.shape) == 3:
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values[0]

            # Render the plot
            st_shap(shap.force_plot(
                float(base_value), # Force to float
                sv,
                input_df.iloc[0],
                feature_names=['Age', 'Income', 'Loan Amt', 'Emp Length', 'Hist Len']
            ))
           
        except Exception as e:
            st.error(f"Visualization Error: {e}")

# --- TAB 2: BATCH PROCESSING (UPDATED WITH PIE CHART) ---
with tab2:
    st.header("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
   
    if uploaded_file:
        st.info(f"Analyzing: {uploaded_file.name}")
        data = pd.read_csv(uploaded_file)
        if all(col in data.columns for col in FEATURE_NAMES):
            # 1. Processing
            batch_scaled = scaler.transform(data[FEATURE_NAMES])
            predictions = model.predict_proba(batch_scaled)[:, 1]
            data['Risk Score (%)'] = (predictions * 100).round(2)
            data['Decision'] = np.where(data['Risk Score (%)'] > 50, 'Reject', 'Approve')
           
            # 2. Key Metrics Row
            st.write("### Global Summary")
            c1, c2, c3 = st.columns(3)
            total_apps = len(data)
            app_rate = round((data['Decision'] == 'Approve').mean() * 100, 1)
            avg_risk = round(data['Risk Score (%)'].mean(), 1)
           
            c1.metric("Total Applications", total_apps)
            c2.metric("Approval Rate", f"{app_rate}%")
            c3.metric("Average Risk Score", f"{avg_risk}%")

            # 3. Visualizations Row
            st.write("---")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.write("#### Approval Breakdown")
                # Create Pie Chart
                decision_counts = data['Decision'].value_counts().reset_index()
                decision_counts.columns = ['Status', 'Count']
               
                fig_pie = go.Figure(data=[go.Pie(
                    labels=decision_counts['Status'],
                    values=decision_counts['Count'],
                    hole=.4, # Makes it a Donut chart for a modern look
                    marker_colors=['#2ecc71', '#e74c3c'] if decision_counts['Status'].iloc[0] == 'Approve' else ['#e74c3c', '#2ecc71']
                )])
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=350)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_chart2:
                st.write("#### Risk Distribution")
                # Updated Histogram
                fig_hist = go.Figure(data=[go.Histogram(x=data['Risk Score (%)'], nbinsx=20, marker_color='teal')])
                fig_hist.update_layout(
                    margin=dict(t=0, b=0, l=0, r=0),
                    height=350,
                    xaxis_title="Risk Score (%)",
                    yaxis_title="Volume"
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # 4. Data Table
            st.write("---")
            st.write("### Detailed Data Preview")
            st.dataframe(data, use_container_width=True)
           
            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Analysis", csv, "batch_risk_analysis.csv", "text/csv")
           
        else:
            st.error(f"Error: CSV file missing required columns: {FEATURE_NAMES}")

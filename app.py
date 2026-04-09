import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIG & ASSETS ---
st.set_page_config(page_title="CreditGuard AI", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    explainer = pickle.load(open('explainer.pkl', 'rb'))
    return model, scaler, explainer

model, scaler, explainer = load_assets()

# --- HEADER ---
st.title("🛡️ CreditGuard AI")
st.markdown("Professional Grade Credit Risk Assessment & Interpretability Portal")
st.divider()

tabs = st.tabs(["👤 Individual Assessment", "📊 Batch Processing"])

# --- TAB 1: INDIVIDUAL ASSESSMENT ---
with tabs[0]:
    with st.container():
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            st.subheader("Customer Profile")
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
            loan = st.number_input("Loan Amount Requested ($)", 0, 500000, 10000)
            score = st.slider("Credit Score", 300, 850, 650)
            emp_years = st.number_input("Years of Employment", 0, 50, 5)
            
            predict_btn = st.button("Generate Risk Report", type="primary", use_container_width=True)

        with col2:
            if predict_btn:
                # Calculations
                features = np.array([[age, income, loan, score, emp_years]])
                features_scaled = scaler.transform(features)
                prob = model.predict_proba(features_scaled)[0][1]
                
                # Financial Metric: Debt-to-Income
                dti = (loan / income) if income > 0 else 0
                
                # 1. GAUGE CHART
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Default Probability (%)", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, 30], 'color': "#2ecc71"},
                            {'range': [30, 70], 'color': "#f1c40f"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # 2. ANALYSIS CARDS
                st.markdown("### Decision Summary")
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    if prob > 0.7:
                        st.error(f"**Risk Status**\n\n HIGH ({prob:.1%})")
                    elif prob > 0.3:
                        st.warning(f"**Risk Status**\n\n MEDIUM ({prob:.1%})")
                    else:
                        st.success(f"**Risk Status**\n\n LOW ({prob:.1%})")

                with m2:
                    # DTI usually risky if > 40%
                    if dti > 0.4:
                        st.error(f"**Loan-to-Income**\n\n {dti:.2%} (High)")
                    else:
                        st.success(f"**Loan-to-Income**\n\n {dti:.2%} (Healthy)")

                with m3:
                    if score < 580:
                        st.error(f"**Score Impact**\n\n Negative ({score})")
                    elif score < 700:
                        st.warning(f"**Score Impact**\n\n Neutral ({score})")
                    else:
                        st.success(f"**Score Impact**\n\n Positive ({score})")

                # 3. CLEAN SHAP EXPLANATION
                st.subheader("🔍 AI Decision Factors")
                st.info("The chart below shows which features pushed the risk score up (Red) or down (Blue).")
                
                shap_values = explainer(features_scaled)
                # Handle SHAP multi-output safely
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                elif len(shap_values.shape) == 3:
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values[0]

                fig_shap, ax = plt.subplots(figsize=(8, 4))
                shap.plots.bar(sv, show=False)
                plt.yticks(range(5), ['Age', 'Income', 'Loan Amt', 'Credit Score', 'Experience'])
                st.pyplot(fig_shap)
            else:
                st.info("← Fill in details and click 'Generate Risk Report' to see the analysis.")

# --- TAB 2: BATCH PROCESSING ---
with tabs[1]:
    uploaded_file = st.file_uploader("Drop Credit Application File (CSV or XLSX)", type=["csv", "xlsx"])
    
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        scaled_batch = scaler.transform(df_input)
        preds_prob = model.predict_proba(scaled_batch)[:, 1]
        
        df_input['Risk_Score'] = preds_prob
        df_input['Decision'] = ["❌ Decline" if p > 0.5 else "✅ Approve" for p in preds_prob]
        
        st.subheader("Batch Results")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.dataframe(df_input.style.background_gradient(subset=['Risk_Score'], cmap='RdYlGn_r'), use_container_width=True)
        with c2:
            fig_pie = px.pie(df_input, names='Decision', color='Decision', 
                             color_discrete_map={'✅ Approve':'#2ecc71','❌ Decline':'#e74c3c'})
            st.plotly_chart(fig_pie, use_container_width=True)

        st.download_button("Download Processed Report", df_input.to_csv(index=False), "Batch_Report.csv", "text/csv")
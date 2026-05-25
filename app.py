"""
app.py  ─  Illness Risk Predictor  (Streamlit)
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import Risk_Prediction as rp

st.set_page_config(
    page_title="Group 1 - Illness Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #ffffff;
    color: #000000;
}

/* Title block */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #111827;  /* FIX: was white → invisible on white background */
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #6b7280;  /* softer gray */
}

/* Metric boxes */
.metric-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}

/* Sidebar (optional: make it white too) */
[data-testid="stSidebar"] {
    background: #f9fafb !important;
}

/* Section headers */
.section-title {
    color: #111827;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #60a5fa) !important;
}
</style>
""", unsafe_allow_html=True)

# Train model once 
@st.cache_resource(show_spinner="Training model on dataset…")
def get_trained_model():
    data_path = os.path.join(os.path.dirname(__file__), "raw_data.csv")
    df = rp.load_and_clean(data_path)
    theta, mean_X, std_X, cost_history, metrics = rp.train(df)
    return theta, mean_X, std_X, cost_history, metrics, df


theta, mean_X, std_X, cost_history, metrics, df_clean = get_trained_model()


# SIDEBAR 
with st.sidebar:
    st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem;'>
            <span style='font-size:2.5rem;'>🩺</span>
            <div style='font-family:"DM Serif Display",serif; font-size:1.3rem; color:#e8edf2; margin-top:0.3rem;'>
                Group1 : Risk_Prediction
            </div>
            <div style='font-size:0.72rem; color:#7a9bbf; letter-spacing:0.08em; text-transform:uppercase;'>
                Illness Risk Prediction Project
            </div>
        </div>
        <hr style='border:none; border-top:1px solid rgba(255,255,255,0.07); margin:1rem 0;'>
        <div style='font-size:0.82rem; color:#7a9bbf; margin-bottom:1.2rem;'>
            Fill in your daily health habits and get an instant risk assessment.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("**Your Profile**")
    age = st.slider("Age", min_value=18, max_value=24, value=20, step=1)

    st.markdown("**Sleep & Hydration**")
    sleep_label = st.select_slider(
        "Sleep Duration (hrs/night)",
        options=['<3', '3-4', '4-5', '5-6', '6-7', '7-8', '>8'],
        value='6-7'
    )
    water_intake = st.slider(
        "Water Intake (litres/day)",
        min_value=0.5, max_value=10.0, value=2.5, step=0.5
    )

    st.markdown("**Lifestyle**")
    stress = st.slider(
        "Stress Level (1 = low, 10 = high)",
        min_value=1, max_value=10, value=5
    )
    screentime = st.slider(
        "Daily Screen Time (hrs)",
        min_value=0.0, max_value=16.0, value=5.0, step=0.5
    )

    st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.07); margin:1rem 0;'>",
                unsafe_allow_html=True)
    predict_btn = st.button("⚡ Predict Risk", use_container_width=True, type="primary")


# MAIN AREA
st.markdown('<div class="badge">Machine Learning using Logistic Regression</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Illness Risk<br><i>Prediction</i></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Prediction", "Data Insights", "Feedback"])


# TAB 1: PREDICTION 
with tab1:

    # Model metrics strip
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, key in zip(
        [m1, m2, m3, m4],
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        ["accuracy", "precision", "recall", "f1_score"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val">{metrics[key]:.0%}</div>
                <div class="metric-lbl">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction result
    st.markdown('<div class="section-title">Your Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    sleep_val = rp.SLEEP_MAP[sleep_label]
    label, prob = rp.predict_single(age, sleep_val, water_intake, stress, screentime)
    prob_pct = prob * 100

    if predict_btn or True:   # always show result after sidebar change
        if label == 1:
            card_cls = "result-high"
            verdict = "⚠️ Higher Risk Detected"
            verdict_color = "#f87171"
            advice = "Consider improving sleep, reducing stress, and staying hydrated."
        else:
            card_cls = "result-low"
            verdict = "✅ Lower Risk Detected"
            verdict_color = "#34d399"
            advice = "Your current habits look healthy — keep it up!"

        col_res, col_gauge = st.columns([3, 2])

        with col_res:
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label" style="color:{verdict_color}">{verdict}</div>
                <div class="result-prob">Illness probability: <strong>{prob_pct:.1f}%</strong></div>
                <div style="margin-top:1rem; font-size:0.9rem; color:#a0b4c8;">{advice}</div>
            </div>""", unsafe_allow_html=True)

            # Input summary
            st.markdown("**Your inputs**")
            summary_data = {
                "Feature": ["Age", "Sleep Duration", "Water Intake", "Stress Level", "Screen Time"],
                "Value": [f"{age} yrs", sleep_label + " hrs", f"{water_intake} L", f"{stress}/10", f"{screentime} hrs"]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        with col_gauge:
            # Probability gauge chart
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')

            theta_range = np.linspace(0, np.pi, 200)
            # background arc
            ax.fill_between(theta_range, 0.65, 1.0,
                             color='#1e2d3d', alpha=0.8)
            # color arc for risk
            fill_up = np.pi * prob
            t_fill = np.linspace(0, fill_up, 200)
            color = "#f87171" if prob >= 0.5 else "#34d399"
            ax.fill_between(t_fill, 0.65, 1.0, color=color, alpha=0.85)

            ax.set_ylim(0, 1)
            ax.set_xlim(0, np.pi)
            ax.set_theta_zero_location("W")
            ax.set_theta_direction(1)
            ax.axis('off')

            ax.text(np.pi / 2, 0.3, f"{prob_pct:.0f}%",
                    ha='center', va='center',
                    fontsize=28, fontweight='bold', color=color,
                    fontfamily='serif')
            ax.text(np.pi / 2, 0.08, "Risk Score",
                    ha='center', va='center',
                    fontsize=10, color='#7a9bbf')

            st.pyplot(fig, use_container_width=True)
            plt.close()

        # Probability bar
        st.markdown("**Risk probability**")
        st.progress(float(prob))

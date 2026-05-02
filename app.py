import streamlit as st
from data.loader import load_data
from models.trainer import train_model
from components.sidebar import render_sidebar
from components.tabs import render_analysis_tab, render_insights_tab, render_dataset_tab
from components.chatbot import render_chatbot_tab
from data.loader import load_data

st.set_page_config(
    page_title="Student Habit Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

df = load_data()
model, r2, mae, X_test, y_test = train_model(df)

u_study, u_sleep, u_phone, analyze_btn = render_sidebar(r2)

st.title("📊 Student Habit Analyzer Pro")
st.caption("Predict your final marks based on your daily habits using Linear Regression.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analysis", "📈 Data Insights", "📁 Dataset", "🤖 AI Chatbot"])

with tab1:
    render_analysis_tab(analyze_btn, model, u_study, u_sleep, u_phone)

with tab2:
    render_insights_tab(df, model)

with tab3:
    render_dataset_tab(df)

with tab4:
    render_chatbot_tab()


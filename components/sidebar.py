import streamlit as st


def render_sidebar(r2: float):
    """Sidebar with model info and input sliders."""
    with st.sidebar:
        st.header("⚙️ Controls")
        st.divider()

        st.info(f"🤖 Model: **Linear Regression**\n\nAccuracy (R²): **{r2 * 100:.1f}%**")

        st.subheader("📋 Your Daily Habits")
        u_study = st.slider("📚 Study Hours",  0.0, 12.0, 5.0, 0.5)
        u_sleep = st.slider("😴 Sleep Hours",  4.0, 10.0, 7.0, 0.5)
        u_phone = st.slider("📱 Phone Usage (hrs)", 0.0, 10.0, 2.0, 0.5)

        total     = u_study + u_sleep + u_phone
        remaining = 24 - total

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Hours Used", f"{total:.1f}")
        col2.metric("Remaining", f"{remaining:.1f}")

        if total > 24:
            st.error("⚠️ Total hours exceed 24! Please reduce.")
            st.stop()

        st.divider()
        analyze_btn = st.button("🚀 Run AI Analysis", use_container_width=True, type="primary")

    return u_study, u_sleep, u_phone, analyze_btn


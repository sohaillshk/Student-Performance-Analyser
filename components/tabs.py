import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from models.trainer import predict, get_grade, FEATURE_COLS

def _simple_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    return fig, ax

def render_analysis_tab(analyze_btn, model, u_study, u_sleep, u_phone):
    if not analyze_btn:
        st.info("👈 Adjust the sliders in the sidebar and click **Run AI Analysis** to see your results.")
        return

    prediction = predict(model, u_study, u_sleep, u_phone)
    grade      = get_grade(prediction)

    st.subheader("📊 Prediction Result")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Predicted Marks", f"{prediction} / 100")
    c2.metric("🏅 Grade",           grade)
    c3.metric("📈 Score Out of 10", f"{prediction / 10:.1f}")

    st.progress(int(prediction), text=f"Performance: {prediction}%")
    st.divider()

    st.subheader("🎯 Goal Setter")
    target = st.slider("Set your target marks:", 50, 100, 85)
    gap    = target - prediction

    if gap > 0:
        extra_study = gap / max(model.coef_[0], 0.1)
        st.warning(
            f"To reach **{target} marks** from your current **{prediction}**, "
            f"try adding approximately **{extra_study:.1f} more study hours/day**."
        )
    else:
        st.success(f"🎉 You're already on track to hit **{target}**! Try setting a higher goal.")

    st.divider()

    st.subheader("💡 Habit Feedback")
    c1, c2, c3 = st.columns(3)

    with c1:
        if u_study < 4:
            st.error("📚 Study hours too low.\nAim for at least 4–5 hrs/day.")
        elif u_study > 9:
            st.warning("📚 Study hours very high.\nAvoid burnout — take breaks.")
        else:
            st.success("📚 Study hours are balanced!")

    with c2:
        if u_sleep < 6:
            st.error("😴 Not enough sleep.\nAim for at least 7 hrs.")
        else:
            st.success("😴 Healthy sleep cycle!")

    with c3:
        if u_phone > 5:
            st.error("📱 High phone usage.\nTry Do Not Disturb mode.")
        elif u_phone > 3:
            st.warning("📱 Moderate phone use.\nUse app timers to stay focused.")
        else:
            st.success("📱 Phone usage is controlled!")


def render_insights_tab(df: pd.DataFrame, model) -> None:
    st.subheader("📈 Statistical Visualizations")

    v1, v2 = st.columns(2)

    with v1:
        st.markdown("**Study Hours vs Marks**")
        fig, ax = _simple_fig()
        sns.regplot(
            data=df, x="Study", y="Marks",
            scatter_kws={"alpha": 0.35, "color": "#4f8ef7"},
            line_kws={"color": "#f97316"},
            ax=ax,
        )
        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Marks")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with v2:
        st.markdown("**Model Feature Weights (Impact per Hour)**")
        weights = pd.Series(model.coef_, index=FEATURE_COLS)
        fig2, ax2 = _simple_fig()
        colors = ["#4f8ef7" if w >= 0 else "#ef4444" for w in weights]
        weights.plot(kind="bar", color=colors, ax=ax2, width=0.5, edgecolor="white")
        ax2.set_xticklabels(FEATURE_COLS, rotation=0)
        ax2.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Coefficient Value")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    st.divider()

    st.markdown("**Marks Distribution Across Students**")
    fig3, ax3 = _simple_fig(w=9, h=3.5)
    ax3.hist(df["Marks"], bins=25, color="#4f8ef7", edgecolor="white", alpha=0.85)
    ax3.axvline(df["Marks"].mean(), color="#f97316", linestyle="--",
                linewidth=1.5, label=f"Mean: {df['Marks'].mean():.1f}")
    ax3.axvline(50, color="#ef4444", linestyle=":", linewidth=1.5, label="Pass Line (50)")
    ax3.set_xlabel("Marks")
    ax3.set_ylabel("Number of Students")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)

    st.divider()

    st.markdown("**Correlation Heatmap**")
    fig4, ax4 = _simple_fig(w=6, h=4)
    sns.heatmap(
        df.corr(), annot=True, fmt=".2f", cmap="YlGn",
        ax=ax4, linewidths=0.5, linecolor="#f0f0f0",
        annot_kws={"size": 9},
    )
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)


def render_dataset_tab(df: pd.DataFrame) -> None:
    st.subheader("📂 Training Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    len(df))
    c2.metric("Average Marks",    f"{df['Marks'].mean():.1f}")
    c3.metric("Top Scorers (≥80)", int((df["Marks"] >= 80).sum()))
    c4.metric("At Risk (<50)",    int((df["Marks"] < 50).sum()))

    st.divider()
    st.dataframe(
        df.style.background_gradient(subset=["Marks"], cmap="YlGn"),
        use_container_width=True,
    )


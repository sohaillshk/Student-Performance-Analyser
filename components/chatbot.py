import streamlit as st
from groq import Groq

SYSTEM_PROMPT = """
You are a friendly and knowledgeable AI Study Advisor built into a Student Habit Analyzer app.

Your job is to answer questions related to:
- Student performance, study habits, and academic improvement
- Effects of sleep, phone usage, and attendance on grades
- Study techniques, time management, and focus strategies
- Understanding ML concepts used in this app (Linear Regression, R² score, etc.)
- General motivation and advice for students

Keep your answers:
- Concise and easy to understand (this is for students)
- Friendly and encouraging in tone
- Practical with real tips whenever possible

If someone asks something completely unrelated to students or education,
politely redirect them back to study-related topics.
"""


def _get_groq_response(api_key: str, history: list, user_msg: str) -> str:
    """Send message to Groq and return the response text."""
    try:
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # free, fast Groq model
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "invalid_api_key" in err.lower() or "authentication" in err.lower():
            return "❌ Invalid API key. Please check your Groq API key and try again."
        if "rate_limit" in err.lower() or "429" in err:
            return "⏳ Rate limit hit. Please wait a few seconds and try again."
        return f"⚠️ Something went wrong: {err}"


def render_chatbot_tab() -> None:
    """Render the full chatbot tab UI."""

    st.subheader("🤖 AI Study Advisor")
    st.caption("Ask me anything about study habits, performance tips, or how this app works!")
    st.divider()

    with st.expander("🔑 Enter your Groq API Key", expanded="groq_api_key" not in st.session_state):
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get your free key at https://console.groq.com/keys",
            label_visibility="collapsed",
        )
        st.caption("🔒 Your key is never stored — only used locally to call Groq.")

        col1, col2 = st.columns([1, 2])
        if col1.button("✅ Save Key", use_container_width=True):
            if api_key.strip():
                st.session_state["groq_api_key"] = api_key.strip()
                st.session_state.pop("chat_history", None)  # reset chat on new key
                st.success("API key saved! Start chatting below ⬇️")
                st.rerun()
            else:
                st.error("Please enter a valid API key.")

        col2.markdown("👉 Get free key: [console.groq.com/keys](https://console.groq.com/keys)")

    if "groq_api_key" not in st.session_state:
        st.info("👆 Please enter your Groq API key above to start chatting.")
        return

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "How many hours should I study daily?",
            "Does sleep really affect my grades?",
            "What is Linear Regression in simple terms?",
            "How can I reduce phone distraction while studying?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, use_container_width=True, key=f"suggestion_{i}"):
                st.session_state["pending_message"] = s
                st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "pending_message" in st.session_state:
        user_input = st.session_state.pop("pending_message")
        _send_message(user_input)

    user_input = st.chat_input("Ask anything about student performance...")
    if user_input:
        _send_message(user_input)

    if st.session_state.get("chat_history"):
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state["chat_history"] = []
            st.rerun()


def _send_message(user_input: str) -> None:
    """Append user message, call Groq, append response, rerun."""
    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history_so_far = st.session_state["chat_history"][:-1]
            response = _get_groq_response(
                st.session_state["groq_api_key"],
                history_so_far,
                user_input,
            )
        st.markdown(response)

    st.session_state["chat_history"].append({"role": "assistant", "content": response})
    st.rerun()


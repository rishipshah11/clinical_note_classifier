import streamlit as st
import requests

st.set_page_config(page_title="Clinical Note Classifier", page_icon="🩺")
st.title("🩺 Clinical Note Classifier")
st.caption("Paste a clinical note to auto-classify its type using ML + NLP")

API_URL = "http://localhost:8000/classify"

if "history" not in st.session_state:
    st.session_state.history = []

note = st.text_area("📝 Enter Clinical Note", height=200,
    placeholder="e.g. Patient prescribed Metformin 500mg twice daily for Type 2 Diabetes...")

col1, col2 = st.columns([1, 4])
with col1:
    classify_btn = st.button("Classify", type="primary")

if classify_btn and note.strip():
    with st.spinner("Analyzing note..."):
        try:
            res = requests.post(API_URL, json={"text": note}).json()
            st.session_state.history.append({
                "note": note[:80] + "...",
                "label": res["label"],
                "emoji": res["emoji"],
                "confidence": res["confidence"],
                "all_classes": res["all_classes"]
            })
            st.success(f"{res['emoji']} **Classified as: {res['label']}**")
            st.metric("Confidence Score", res["confidence"])
            with st.expander("📊 All Class Scores"):
                for cls, score in sorted(res["all_classes"].items(), key=lambda x: -x[1]):
                    st.write(f"`{cls}` → {round(score, 3)}")
        except Exception as e:
            st.error(f"API Error: {e}. Make sure FastAPI is running.")

elif classify_btn:
    st.warning("Please enter a clinical note first.")

if st.session_state.history:
    st.divider()
    st.subheader("📋 Classification History")
    for i, item in enumerate(reversed(st.session_state.history)):
        st.write(f"{item['emoji']} **{item['label']}** — _{item['note']}_")

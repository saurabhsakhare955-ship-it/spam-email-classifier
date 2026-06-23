import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import re

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #e53935; text-align: center; }
    .subtitle   { font-size: 1rem; color: #555; text-align: center; margin-bottom: 1rem; }
    .spam-result { background: #ffebee; border-left: 5px solid #e53935;
                   padding: 1rem 1.5rem; border-radius: 5px; }
    .ham-result  { background: #e8f5e9; border-left: 5px solid #43a047;
                   padding: 1rem 1.5rem; border-radius: 5px; }
    .keyword-tag { display: inline-block; background: #fff3e0; border: 1px solid #ff9800;
                   border-radius: 12px; padding: 2px 10px; margin: 3px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📧 Spam Email Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">TF-IDF + SVM | Naive Bayes | ~98% Accuracy</p>', unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("model/spam_model.pkl"):
        st.error("❌ Model not found! Run `python train_model.py` first.")
        st.stop()
    model      = pickle.load(open("model/spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()
st.success("✅ Model loaded!")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("""
    **Algorithm:** LinearSVC (Support Vector Machine)

    **Feature Extraction:** TF-IDF
    - Max features: 5,000
    - N-grams: (1, 2)
    - Stop words: English

    **Dataset:** SMS Spam Collection
    - ~5,572 messages
    - 87% Ham / 13% Spam

    **Accuracy:** ~98%
    **Precision (Spam):** ~99%
    **Recall (Spam):** ~95%
    """)

    if os.path.exists("model/results.png"):
        st.image("model/results.png", caption="Model Results")

# ── Spam Examples ──────────────────────────────────────────────────────────────
SPAM_EXAMPLES = [
    "WINNER!! You've won a FREE iPhone! Click here NOW to claim your prize! Limited offer!",
    "Congratulations! You've been selected for a $1,000,000 prize! Call 0800-FREE-GIFT to claim!",
    "URGENT: Your account will be suspended! Verify now at secure-login.fake.com immediately!",
    "FREE ringtones! Txt RINGTONE to 87070. £1.50/msg. Ends 30th June. 18+ only.",
    "Make $5000 per week from home! No experience needed! 100% guaranteed income!",
]

HAM_EXAMPLES = [
    "Hey, are we still meeting for lunch tomorrow at the usual place around noon?",
    "Could you please send me the quarterly report before end of day? Thanks!",
    "Don't forget Mom's birthday dinner is this Saturday at 7pm. See you there!",
    "I'll be working from home today. Feel free to call if you need anything.",
    "The project deadline has been moved to next Friday. Let me know if you have concerns.",
]

# ── Main Area ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("✉️ Enter Email / Message Text")

    # Quick example buttons
    st.markdown("**Quick Examples:**")
    ex_col1, ex_col2 = st.columns(2)
    
    selected_text = ""
    with ex_col1:
        if st.button("🚨 Spam Example 1"):
            selected_text = SPAM_EXAMPLES[0]
        if st.button("🚨 Spam Example 2"):
            selected_text = SPAM_EXAMPLES[1]
        if st.button("🚨 Spam Example 3"):
            selected_text = SPAM_EXAMPLES[2]

    with ex_col2:
        if st.button("✅ Ham Example 1"):
            selected_text = HAM_EXAMPLES[0]
        if st.button("✅ Ham Example 2"):
            selected_text = HAM_EXAMPLES[1]
        if st.button("✅ Ham Example 3"):
            selected_text = HAM_EXAMPLES[2]

    # Text area
    email_text = st.text_area(
        "Type or paste your message below:",
        value=selected_text,
        height=180,
        placeholder="Enter email or SMS text here..."
    )

    classify_btn = st.button("🔍 Classify Message", type="primary", use_container_width=True)

with col_right:
    st.subheader("📈 Spam Indicators")
    st.markdown("""
    **Common spam signals:**
    - ALL CAPS words
    - Exclamation marks!!!
    - Prize / Winner / FREE
    - Urgency (ACT NOW, LIMITED)
    - Phone numbers / URLs
    - Money amounts ($$$)
    - Click here / Download
    """)

# ── Classification Result ──────────────────────────────────────────────────────
if classify_btn and email_text.strip():
    st.markdown("---")
    st.subheader("🎯 Classification Result")

    # Predict
    text_vec   = vectorizer.transform([email_text])
    prediction = model.predict(text_vec)[0]

    # Feature analysis
    feature_names = vectorizer.get_feature_names_out()
    text_arr      = text_vec.toarray()[0]
    top_indices   = text_arr.argsort()[-15:][::-1]
    top_keywords  = [(feature_names[i], round(float(text_arr[i]), 4))
                     for i in top_indices if text_arr[i] > 0]

    # Basic text stats
    word_count  = len(email_text.split())
    char_count  = len(email_text)
    caps_count  = sum(1 for w in email_text.split() if w.isupper() and len(w) > 1)
    excl_count  = email_text.count('!')
    has_url     = bool(re.search(r'http[s]?://|www\.', email_text, re.I))
    has_phone   = bool(re.search(r'\b\d{10,}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', email_text))
    has_money   = bool(re.search(r'\$[\d,]+|£[\d,]+|€[\d,]+|\b\d+\s*(dollars?|pounds?)\b', email_text, re.I))

    r1, r2 = st.columns(2)

    with r1:
        if prediction == 1:
            st.markdown("""
            <div class="spam-result">
                <h2 style="color:#e53935; margin:0">🚨 SPAM DETECTED</h2>
                <p style="margin:0.5rem 0 0 0; color:#555">This message is classified as <strong>SPAM</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ham-result">
                <h2 style="color:#43a047; margin:0">✅ HAM (Legitimate)</h2>
                <p style="margin:0.5rem 0 0 0; color:#555">This message is classified as <strong>LEGITIMATE</strong></p>
            </div>
            """, unsafe_allow_html=True)

    with r2:
        st.markdown("**📊 Text Statistics:**")
        stats_df = pd.DataFrame({
            "Feature": ["Word Count", "Char Count", "CAPS Words", "Exclamations (!)", "Has URL", "Has Phone", "Has Money"],
            "Value":   [word_count, char_count, caps_count, excl_count, "Yes" if has_url else "No",
                        "Yes" if has_phone else "No", "Yes" if has_money else "No"]
        })
        st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # Keywords
    if top_keywords:
        st.markdown("---")
        st.subheader("🔑 Top Keywords Found (by TF-IDF Score)")
        kw_df = pd.DataFrame(top_keywords, columns=["Keyword/Phrase", "TF-IDF Score"])
        
        col_kw1, col_kw2 = st.columns([1, 2])
        with col_kw1:
            st.dataframe(kw_df, hide_index=True, use_container_width=True)
        with col_kw2:
            st.bar_chart(
                pd.DataFrame({"TF-IDF Score": [k[1] for k in top_keywords]},
                             index=[k[0] for k in top_keywords])
            )

elif classify_btn:
    st.warning("⚠️ Please enter some text to classify!")

# ── Bulk Classification ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📂 Bulk Classification (Upload CSV)"):
    st.markdown("Upload a CSV file with a column named `message` to classify multiple emails at once.")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_csv:
        try:
            bulk_df = pd.read_csv(uploaded_csv)
            if "message" not in bulk_df.columns:
                st.error("❌ CSV must have a column named 'message'")
            else:
                bulk_vecs = vectorizer.transform(bulk_df["message"].fillna(""))
                bulk_preds = model.predict(bulk_vecs)
                bulk_df["prediction"] = ["SPAM 🚨" if p == 1 else "HAM ✅" for p in bulk_preds]
                
                spam_count = sum(bulk_preds)
                ham_count  = len(bulk_preds) - spam_count
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Messages", len(bulk_preds))
                c2.metric("Spam Detected", spam_count)
                c3.metric("Legitimate (Ham)", ham_count)
                
                st.dataframe(bulk_df[["message", "prediction"]], use_container_width=True)
                
                csv_out = bulk_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results CSV", csv_out, "classified_emails.csv", "text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
    Built with Scikit-learn & Streamlit | Codec Technologies AI Internship Project
</div>
""", unsafe_allow_html=True)

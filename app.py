import streamlit as st
import joblib
import pandas as pd
import requests
import base64
from bs4 import BeautifulSoup
from urllib.parse import quote
import time
from collections import Counter
import re



# LOAD BOTH MODELS
try:
    model_old = joblib.load("fake_news_model.pkl")
    vectorizer_old = joblib.load("tfidf_vectorizer.pkl")
    model_new = joblib.load("new_fake_news_model.pkl")
    vectorizer_new = joblib.load("new_tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading models. Please ensure .pkl files are present. Details: {e}")
    st.stop()



# PAGE CONFIG
st.set_page_config(page_title="Fake News Detection System", layout="wide")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📰 Quick Test"


# IMAGE LOADER
def load_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""


brain_image_base64 = load_image_base64("ai_image.gif")
brain_image_html = (
    f'<img src="data:image/png;base64,{brain_image_base64}" width="250">'
    if brain_image_base64 else ""
)



# CUSTOM CSS
st.markdown("""
<style>
.hero-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3rem 2rem 1rem 2rem;
    border-radius: 12px;
    background-color: #f5f5f5;
    margin-bottom: 30px;
}
.hero-text { max-width: 55%; }
.hero-title { font-size: 3.2rem; font-weight: 800; margin-bottom: 0.8rem; line-height: 1.2; }
.hero-sub { font-size: 1.1rem; color: #444; margin-bottom: 1.5rem; max-width: 90%; line-height: 1.5; }
.badge-row { margin-top: 15px; }
.badge { display:inline-block; background-color:#111827; color:white; padding:6px 14px; border-radius:20px; margin-right:8px; margin-top:5px; font-size:0.85rem; font-weight:500; }

.stTabs [role="tab"] {
    background-color:#f8f8f8; color:black; padding:10px; border:1px solid #ccc;
    border-radius:6px 6px 0 0; margin-right:6px; font-weight:500;
}
.stTabs [role="tab"]:hover { background-color:#e5e5e5; }
.stTabs [aria-selected="true"] { background-color:#111827!important; color:white!important; }

textarea:hover, .stTextInput>div>div>input:hover {
    border:1px solid #111!important; box-shadow:0 0 0 2px #00000022;
}
</style>
""", unsafe_allow_html=True)



# HERO SECTION
with st.container():
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-text">
            <div class="hero-title">A new standard for <br> information integrity.</div>
            <div class="hero-sub">
                Powered by machine learning and trained on <strong>40,000+ news articles</strong> plus the <strong>LIAR benchmark</strong>,
                this system detects misinformation with <strong>98.88%+ accuracy</strong> using Logistic Regression models, advanced
                <strong>TF-IDF vectorization</strong>, and intelligent <strong>NLP pre-processing</strong>.
                <br><br>Compare models, analyze trends, and stop fake news—smarter than ever.
            </div>
            <div class="badge-row">
                <span class="badge">Logistic Regression</span>
                <span class="badge">New Improved Model</span>
                <span class="badge">LIAR Dataset</span>
                <span class="badge">TF-IDF</span>
                <span class="badge">Scikit‑Learn</span>
            </div>
        </div>
        <div>{brain_image_html}</div>
    </div>
    """, unsafe_allow_html=True)


# METRICS
col1, col2, col3 = st.columns(3)
col1.metric("📦 Dataset Size", "40,000+ Articles")
col2.metric("🎯 Highest Accuracy", "98.88%")
col3.metric("🧠 Models Loaded", "Multiple Logistic Regression Models")

progress = st.progress(0)
for i in range(0, 101, 5):
    time.sleep(0.005)
    progress.progress(i)


# FACT CHECK TOGGLE
use_fact_check = st.sidebar.checkbox("🔎 Enable Fact Check Integration", value=True)



# PREDICTION FUNCTIONS
def predict_old(text: str):
    vec = vectorizer_old.transform([text])
    pred = model_old.predict(vec)[0]
    prob = float(model_old.predict_proba(vec)[0].max())
    label = "🟢 REAL" if int(pred) == 1 else "🔴 FAKE"
    return label, prob


def predict_new(text: str):
    vec = vectorizer_new.transform([text])
    pred = model_new.predict(vec)[0]
    prob = float(model_new.predict_proba(vec)[0].max())
    label = "🟢 REAL" if int(pred) == 1 else "🔴 FAKE"
    return label, prob


def scrape_factcheck(query: str):
    url = f"https://toolbox.google.com/factcheck/explorer/search/{query}"
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        snippets = soup.find_all("div", class_="snippet-text")
        return [s.get_text(strip=True) for s in snippets][:3]
    except Exception:
        return []


# TABS
tabs = ["📰 Quick Test", "📊 Data Scientist", "📡 PR Analyst", "🎓 Professor", "📣 Moderator"]
tab_objects = st.tabs(tabs)




# 1 DETECTOR TAB (New Model)
with tab_objects[0]:
    st.header("📰 News Article Classifier")

    txt = st.text_area("Paste article text:", height=260)

    if st.button("🔍 Classify Now"):
        if txt.strip():
            label, conf = predict_new(txt)
            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf:.2%}")

            if use_fact_check:
                q = quote(" ".join(txt.split()[:12]))
                st.markdown(f"🔗 [Open Fact Check](https://toolbox.google.com/factcheck/explorer/search/{q})")
                facts = scrape_factcheck(q)
                for i, ftxt in enumerate(facts, 1):
                    st.markdown(f"**{i}.** {ftxt}")
        else:
            st.warning("Enter article text!")




# 2 DATA SCIENTIST TAB (Dashboard + Batch + Trends, New Model)
with tab_objects[1]:
    st.header("📊 Data Scientist – Model Insights & Tools")
    st.caption("(Analyze Content Trends)")

    # LIVE INFERENCE CONSOLE (New Model)
    st.markdown("---")
    st.subheader("💻 Live Inference Console (New Model)")

    test_input = st.text_area(
        "Enter text for model testing:",
        "Breaking: Scientists discover water on the sun.",
        height=120
    )

    if st.button("▶ Run Inference", key="inference_ds"):
        label, conf = predict_new(test_input)
        st.code(f"""
# INPUT
{test_input}

# OUTPUT
Prediction → {label}
Confidence → {conf:.2%}
""", language="bash")

    # BATCH CLASSIFICATION
    st.markdown("---")
    st.subheader("📦 Batch Classification (Upload CSV)")

    file = st.file_uploader("Upload CSV with 'text' column", type=["csv"], key="ds")

    if file:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column.")
        else:
            with st.spinner("Classifying articles with New Model..."):
                texts = df["text"].fillna("").astype(str)
                vecs = vectorizer_new.transform(texts)
                preds = model_new.predict(vecs)
                confs = model_new.predict_proba(vecs).max(axis=1)

            df["Prediction"] = ["🟢 REAL" if int(p) == 1 else "🔴 FAKE" for p in preds]
            df["Confidence"] = [f"{float(c):.2%}" for c in confs]

            st.dataframe(df)

            # DOWNLOAD RESULTS
            csv_processed = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Classified Batch Results",
                data=csv_processed,
                file_name="classified_batch_results.csv",
                mime="text/csv"
            )

            # TREND ANALYTICS
            st.markdown("---")
            st.subheader("📈 Trend Analytics – Fake Content Patterns")

            # 1 Fake vs Real distribution
            st.markdown("### 1️⃣ Fake vs Real Distribution")
            trend_counts = df["Prediction"].value_counts()
            st.bar_chart(trend_counts)

            # 2 Article length trend
            st.markdown("### 2️⃣ Article Length Trend")
            df["length"] = df["text"].astype(str).apply(len)
            length_stats = df.groupby("Prediction")["length"].mean()

            col_len1, col_len2 = st.columns(2)
            with col_len1:
                st.write("**Average Length by Class:**")
                st.dataframe(length_stats)

            with col_len2:
                st.write("**Length Distribution (all articles):**")
                st.line_chart(df[["length"]])

            # 3 Keyword trend
            st.markdown("### 3️⃣ Top Keywords in Fake Articles")
            fake_texts = " ".join(df[df["Prediction"] == "🔴 FAKE"]["text"].astype(str))
            words = re.findall(r"\b[a-zA-Z]{4,}\b", fake_texts.lower())
            common_words = Counter(words).most_common(10)
            kw_df = pd.DataFrame(common_words, columns=["Keyword", "Frequency"])
            st.dataframe(kw_df)

            # Summary insights
            st.markdown("### 🧠 Trend Summary (Auto-Generated Insights)")
            total_fake = int((df["Prediction"] == "🔴 FAKE").sum())
            total_real = int((df["Prediction"] == "🟢 REAL").sum())
            total = total_fake + total_real if (total_fake + total_real) > 0 else 1
            top_kw = kw_df.iloc[0]["Keyword"] if not kw_df.empty else "N/A"
            fake_avg_len = float(length_stats.get("🔴 FAKE", 0.0))

            summary = f"""
- **Fake Articles:** {total_fake}
- **Real Articles:** {total_real}
- **Fake Article Share:** {total_fake / total:.2%}
- **Most Frequent Fake Keyword:** `{top_kw}`
- **Fake Articles Avg Length:** {fake_avg_len:.1f} characters
            """
            st.info(summary)



# 3 PR ANALYST TAB
with tab_objects[2]:
    st.header("📡 PR Analyst – Misinformation Monitor")


    st.markdown("### 🔍 Filter Settings")
    keywords = st.text_input("Brand Keywords (comma separated):", placeholder="e.g. Amazon, Tesla, Crisis")

    tab_pr_method = st.radio("Input Method:", ["📂 Upload CSV", "✍️ Paste Text"])

    # Method 1: Paste Text
    if tab_pr_method == "✍️ Paste Text":
        pr_text = st.text_area("Paste content to check against keywords:")
        if st.button("Check Content"):
            if not keywords:
                st.warning("Please enter keywords first.")
            elif not pr_text:
                st.warning("Please paste text.")
            else:
                keys = [k.strip().lower() for k in keywords.split(",") if k.strip()]
                found_keys = [k for k in keys if k in pr_text.lower()]

                if found_keys:
                    st.success(f"Keywords Found: {', '.join(found_keys)}")
                    label, conf = predict_new(pr_text)
                    st.metric("Classification", label)
                    st.metric("Risk Score", f"{conf:.2%}")
                else:
                    st.info("No brand keywords found in this text.")

    # Method 2: CSV Upload
    elif tab_pr_method == "📂 Upload CSV":
        pr_file = st.file_uploader("Upload dataset with 'text' column", type=["csv"], key="pr_upload")

        if pr_file and keywords:
            df = pd.read_csv(pr_file)
            if "text" in df.columns:
                keys = [k.strip().lower() for k in keywords.split(",") if k.strip()]


                def match(x: str) -> bool:
                    s = str(x).lower()
                    return any(k in s for k in keys)


                # Filter by brand keywords
                subset = df[df["text"].apply(match)].copy()

                if subset.empty:
                    st.warning("No matching articles found containing those keywords.")
                else:
                    with st.spinner("Classifying filtered articles with New Model..."):
                        texts = subset["text"].fillna("").astype(str)
                        vecs = vectorizer_new.transform(texts)
                        preds = model_new.predict(vecs)
                        confs = model_new.predict_proba(vecs).max(axis=1)

                    subset["Prediction"] = ["🟢 REAL" if int(p) == 1 else "🔴 FAKE" for p in preds]
                    subset["Confidence"] = [f"{float(c):.2%}" for c in confs]

                    st.subheader("🚩 Flagged Articles Report (All)")
                    st.dataframe(subset)

                    st.markdown("---")

                    # MITIGATION DOWNLOAD ---
                    col_download_all, col_download_fake = st.columns(2)

                    with col_download_all:
                        csv_all = subset.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Full Report (Real & Fake)",
                            data=csv_all,
                            file_name="brand_monitoring_full.csv",
                            mime="text/csv"
                        )

                    with col_download_fake:
                        # FILTER ONLY FAKE
                        df_fake_only = subset[subset["Prediction"] == "🔴 FAKE"]

                        if not df_fake_only.empty:
                            csv_fake = df_fake_only.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⚠️ Download Flagged Mitigation Report (FAKE ONLY)",
                                data=csv_fake,
                                file_name="brand_mitigation_targets.csv",
                                mime="text/csv",
                                help="Downloads only the fake articles so you can take action (mitigate)."
                            )
                        else:
                            st.success("✅ No fake articles found for this brand! No mitigation needed.")
                    # ----------------------------------------------------

            else:
                st.error("Dataset must contain 'text' column.")



# 4 PROFESSOR TAB
with tab_objects[3]:
    st.header("🎓 Professor – Model Comparison Lab")

    st.markdown("Compare predictions from the Old Model and the New LIAR-augmented Model.")

    comparison_text = st.text_area("Enter article text for comparison:", height=150)

    if st.button("Compare Models"):
        if comparison_text.strip():

            old_label, old_conf = predict_old(comparison_text)
            new_label, new_conf = predict_new(comparison_text)

            col_old, col_new = st.columns(2)

            with col_old:
                st.subheader("Old Model")
                st.write(f"Prediction: **{old_label}**")
                st.write(f"Confidence: **{old_conf:.2%}**")

            with col_new:
                st.subheader("New Model")
                st.write(f"Prediction: **{new_label}**")
                st.write(f"Confidence: **{new_conf:.2%}**")

            if old_label != new_label:
                st.error("Models disagree on this article.")
            else:
                st.success("Both models agree on this prediction.")

            # CSV DOWNLOAD
            report_df = pd.DataFrame({
                "Model": ["Old Model", "New Model"],
                "Prediction": [old_label, new_label],
                "Confidence": [f"{old_conf:.2%}", f"{new_conf:.2%}"]
            })
            csv_data = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Comparison Report (CSV)",
                data=csv_data,
                file_name="model_comparison_report.csv",
                mime="text/csv"
            )

            # FEATURE LOGS
            st.markdown("---")
            st.subheader("🛠 Technical Logs & Feature Extraction")
            with st.expander("View Feature Extraction Logs (Top TF-IDF Terms)"):
                st.write("These terms had the highest weight in the New Model's decision:")
                try:
                    feature_names = vectorizer_new.get_feature_names_out()
                    doc_vec = vectorizer_new.transform([comparison_text])
                    nz_indices = doc_vec.nonzero()[1]
                    features = [(feature_names[i], doc_vec[0, i]) for i in nz_indices]
                    features.sort(key=lambda x: x[1], reverse=True)

                    top_features = features[:20]
                    feat_df = pd.DataFrame(top_features, columns=["Feature (Word)", "TF-IDF Weight"])
                    st.dataframe(feat_df)
                except Exception as e:
                    st.error(f"Could not extract features: {e}")

        else:
            st.warning("Enter text first.")



# 5 MODERATOR TAB
with tab_objects[4]:
    st.header("📣 Moderator – Content Verification")

    mode = st.radio(
        "Select verification type:",
        ["📰 Check Headlines", "📝 Check Article"]
    )


    # HEADLINE CHECKER
    if mode == "📰 Check Headlines":
        st.subheader("📰 Headline Checker")

        raw = st.text_area(
            "Enter headlines (one per line):",
            placeholder="Example:\nGovernment announces new policy\nAliens land in New York",
            height=180
        )

        if st.button("Check Headlines"):
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one headline.")
            else:
                results = []
                for l in lines:
                    label, conf = predict_new(l)
                    results.append((l, label, conf))

                df_mod = pd.DataFrame(
                    results,
                    columns=["Headline", "Prediction", "Confidence"]
                )
                st.dataframe(df_mod)


    #ARTICLE CHECKER
    elif mode == "📝 Check Article":
        st.subheader("📝 Article Checker")

        article_text = st.text_area(
            "Paste full news article text:",
            placeholder="Paste the complete article here...",
            height=260
        )

        if st.button("Check Article"):
            if not article_text.strip():
                st.warning("Please paste article text.")
            else:
                label, conf = predict_new(article_text)
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {conf:.2%}")

                if use_fact_check:
                    q = quote(" ".join(article_text.split()[:12]))
                    st.markdown(
                        f"🔗 [Open Fact Check]"
                        f"(https://toolbox.google.com/factcheck/explorer/search/{q})"
                    )
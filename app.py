import streamlit as st
import joblib, re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARTICLE_MODEL = BASE / "fake_news_model.joblib"
HEADLINE_MODEL = BASE / "headline_model.joblib"

st.set_page_config(
    page_title="Nigerian Fake News Detection System",
    page_icon="📰",
    layout="centered",
)

@st.cache_resource
def load_models():
    article = joblib.load(ARTICLE_MODEL)
    headline = joblib.load(HEADLINE_MODEL)
    return article, headline

def clean(s):
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)
    s = re.sub(r"#(\w+)", r"\1", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

article_bundle, headline_bundle = load_models()

st.title("Fake News Detection System")
st.write(
    "Natural Language Processing prototype for classifying news text "
    "as potentially fake or real."
)

mode = st.radio(
    "What are you entering?",
    ["Headline", "Full article"],
    horizontal=True,
)

if mode == "Headline":
    st.info(
        "Headline mode uses a model trained specifically on WELFake headlines. "
        "For longer evidence, use Full article mode."
    )
    prompt = "Enter a news headline"
    height = 120
    bundle = headline_bundle
else:
    st.info(
        "Full article mode uses the article model. Paste the article text "
        "(including its headline when available)."
    )
    prompt = "Enter the full news article"
    height = 260
    bundle = article_bundle

text = st.text_area(prompt, height=height)

if st.button("Analyse", type="primary"):
    cleaned = clean(text)

    if not cleaned:
        st.warning("Please enter text to analyse.")
    else:
        token_count = len(cleaned.split())

        # Avoid pretending that extremely tiny inputs are reliable.
        if mode == "Headline" and token_count < 3:
            st.warning("Please enter a complete headline with at least 3 words.")
        elif mode == "Full article" and token_count < 20:
            st.warning(
                "Please paste a fuller article. Very short text is outside the "
                "main evaluation condition of the article model."
            )
        else:
            vec = bundle["vectorizer"]
            clf = bundle["classifier"]
            X = vec.transform([cleaned])
            prediction = int(clf.predict(X)[0])
            probabilities = clf.predict_proba(X)[0]

            # Do not assume probability column order; use classifier.classes_.
            class_probs = {
                int(cls): float(prob)
                for cls, prob in zip(clf.classes_, probabilities)
            }
            fake_prob = class_probs.get(0, 0.0)
            real_prob = class_probs.get(1, 0.0)
            confidence = max(fake_prob, real_prob)

            if prediction == 0:
                st.error(
                    f"Prediction: POTENTIALLY FAKE / MISLEADING  | "
                    f"Model confidence: {confidence:.1%}"
                )
            else:
                st.success(
                    f"Prediction: POTENTIALLY REAL  | "
                    f"Model confidence: {confidence:.1%}"
                )

            st.write(
                f"Fake probability: **{fake_prob:.1%}**  \n"
                f"Real probability: **{real_prob:.1%}**"
            )

            st.caption(
                "Important: this is automated text classification, not independent "
                "fact-checking. A prediction is not proof that a claim is true or false. "
                "Verify important claims using authoritative sources."
            )

with st.expander("About the system"):
    st.write(
        "The system uses text preprocessing, hashed word features and a "
        "logistic-loss linear classifier. Headline mode uses a model trained "
        "on WELFake headlines; Full article mode uses the article model."
    )

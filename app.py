import streamlit as st
import joblib, re
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL = BASE / "fake_news_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL)

def clean(s):
    s=str(s).lower()
    s=re.sub(r"https?://\S+|www\.\S+"," URL ",s)
    s=re.sub(r"@\w+"," USER ",s)
    s=re.sub(r"#(\w+)",r"\1",s)
    s=re.sub(r"[^a-z0-9\s']"," ",s)
    return re.sub(r"\s+"," ",s).strip()

bundle=load_model()
vec, clf=bundle["vectorizer"], bundle["classifier"]

st.set_page_config(page_title="Nigerian Fake News Detection System", page_icon="📰")
st.title("Fake News Detection System")
st.write("Natural Language Processing prototype for classifying news text as potentially fake or real.")

text=st.text_area("Enter a news headline or article", height=220)

if st.button("Analyse", type="primary"):
    if not text.strip():
        st.warning("Please enter text to analyse.")
    else:
        prediction=int(clf.predict(vec.transform([clean(text)]))[0])
        if hasattr(clf, "predict_proba"):
            confidence=float(max(clf.predict_proba(vec.transform([clean(text)]))[0]))
        else:
            score=float(clf.decision_function(vec.transform([clean(text)]))[0])
            confidence=float(1/(1+__import__("math").exp(-abs(score))))
        if prediction==0:
            st.error(f"Prediction: POTENTIALLY FAKE / MISLEADING  |  Model confidence: {confidence:.1%}")
        else:
            st.success(f"Prediction: POTENTIALLY REAL  |  Model confidence: {confidence:.1%}")
        st.caption("Important: this is automated text classification, not independent fact-checking. Verify important claims using authoritative sources.")

with st.expander("About the system"):
    st.write("The model uses text preprocessing, hashed word features and a logistic-loss linear classifier trained on the WELFake dataset.")

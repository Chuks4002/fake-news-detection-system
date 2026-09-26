
import streamlit as st
import joblib, re, os, requests
from pathlib import Path
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

BASE = Path(__file__).resolve().parent
ARTICLE_MODEL = BASE / "fake_news_model.joblib"
HEADLINE_MODEL = BASE / "headline_model.joblib"

st.set_page_config(page_title="Nigerian Fake News Detection System", page_icon="📰")

@st.cache_resource
def load_models():
    return joblib.load(ARTICLE_MODEL), joblib.load(HEADLINE_MODEL)

def clean(s):
    s = str(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USER ", s)
    s = re.sub(r"#(\w+)", r"\1", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def model_prediction(bundle, text):
    vec, clf = bundle["vectorizer"], bundle["classifier"]
    X = vec.transform([clean(text)])
    pred = int(clf.predict(X)[0])
    probs = clf.predict_proba(X)[0]
    p = {int(c): float(v) for c, v in zip(clf.classes_, probs)}
    return pred, p.get(0, 0.0), p.get(1, 0.0)

def google_fact_checks(claim):
    key = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", os.getenv("GOOGLE_FACTCHECK_API_KEY", ""))
    if not key:
        return [], "no_api_key"
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"key": key, "query": claim, "languageCode": "en", "pageSize": 10}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json().get("claims", []), "ok"
    except Exception as e:
        return [], f"error:{e}"

def web_evidence_search(claim):
    """Discovery-only fallback. It does not invent a truth verdict."""
    q = quote_plus(claim + " fact check")
    url = f"https://html.duckduckgo.com/html/?q={q}"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for item in soup.select(".result")[:5]:
            a = item.select_one(".result__a")
            sn = item.select_one(".result__snippet")
            if a:
                results.append({
                    "title": a.get_text(" ", strip=True),
                    "url": a.get("href", ""),
                    "snippet": sn.get_text(" ", strip=True) if sn else ""
                })
        return results
    except Exception:
        return []

def normalise_rating(rating):
    r = rating.lower()
    if any(x in r for x in ["false", "fake", "incorrect", "wrong", "misleading", "pants on fire"]):
        return "Potentially Fake / Misleading"
    if any(x in r for x in ["true", "correct", "accurate"]):
        return "Potentially Real"
    if any(x in r for x in ["unproven", "unverified", "unclear", "unsupported"]):
        return "Unverified"
    return "Unverified"

def render_fact_checks(claims):
    if not claims:
        return None
    # Show the closest available reviews, preserving publisher attribution.
    reviews = []
    for c in claims[:5]:
        for cr in c.get("claimReview", []):
            reviews.append({
                "claim": c.get("text", ""),
                "publisher": cr.get("publisher", {}).get("name", "Unknown publisher"),
                "rating": cr.get("textualRating", "Unspecified"),
                "url": cr.get("url", "")
            })
    return reviews

article_bundle, headline_bundle = load_models()

st.title("Fake News Detection & Claim Verification System")
st.write("NLP classification combined with evidence-based claim verification.")

st.info(
    "The NLP model detects patterns learned from labelled news data. "
    "The verification layer checks whether the claim has published fact-check evidence. "
    "A claim is never called true merely because the NLP classifier predicts REAL."
)

demo = st.selectbox(
    "Demo examples",
    [
        "Choose a demo…",
        "The sky is red",
        "Nigeria has 36 states",
    ],
)
if demo != "Choose a demo…":
    st.session_state["claim_text"] = demo

mode = st.radio("Input type", ["Headline / claim", "Full article"], horizontal=True)
claim = st.text_area(
    "Enter a claim or news text",
    value=st.session_state.get("claim_text", ""),
    height=220 if mode == "Full article" else 130,
)

if st.button("Analyse & Verify", type="primary"):
    text = claim.strip()
    if not text:
        st.warning("Please enter text to analyse.")
        st.stop()

    bundle = headline_bundle if mode == "Headline / claim" else article_bundle
    if mode == "Headline / claim" and len(clean(text).split()) < 3:
        st.warning("Please enter a complete claim or headline.")
        st.stop()
    if mode == "Full article" and len(clean(text).split()) < 20:
        st.warning("For article mode, paste a fuller article.")
        st.stop()

    pred, fake_p, real_p = model_prediction(bundle, text)

    with st.spinner("Checking available fact-check evidence…"):
        claims, status = google_fact_checks(text)
        reviews = render_fact_checks(claims)
        discovery = [] if reviews else web_evidence_search(text)

    st.subheader("Verification result")

    if reviews:
        # Use attributed fact-check publisher ratings as the evidence verdict.
        primary = reviews[0]
        verdict = normalise_rating(primary["rating"])
        if verdict == "Potentially Fake / Misleading":
            st.error(f"VERDICT: {verdict}")
        elif verdict == "Potentially Real":
            st.success(f"VERDICT: {verdict}")
        else:
            st.warning(f"VERDICT: {verdict}")

        st.write(
            f"**Fact-check publisher:** {primary['publisher']}  \n"
            f"**Published rating:** {primary['rating']}"
        )
        if primary["url"]:
            st.link_button("Open fact-check", primary["url"])

        st.caption(
            "The verdict above is attributed to the fact-check publisher. "
            "It is not generated by the NLP classifier."
        )
    else:
        st.warning("VERDICT: UNVERIFIED")
        if status == "no_api_key":
            st.write(
                "No Google Fact Check Tools API key is configured, so the app "
                "cannot query the indexed fact-check database yet."
            )
        else:
            st.write(
                "No matching published fact-check was found. This does not mean "
                "the claim is true; it means the system does not have sufficient "
                "verified evidence to issue a fake/real verdict."
            )

        if discovery:
            st.markdown("**Evidence candidates found on the web:**")
            for item in discovery:
                st.markdown(f"**{item['title']}**")
                if item["snippet"]:
                    st.caption(item["snippet"])
                if item["url"]:
                    st.link_button("Open source", item["url"])
        else:
            st.caption("No evidence candidates were retrieved.")

    st.subheader("NLP model assessment")
    label = "Potentially Fake / Misleading" if pred == 0 else "Potentially Real"
    st.write(f"**NLP classification:** {label}")
    st.write(f"Fake probability: **{fake_p:.1%}**")
    st.write(f"Real probability: **{real_p:.1%}**")

    st.caption(
        "Important: NLP classification and factual verification are separate. "
        "The classifier does not establish truth. When verified evidence is unavailable, "
        "the system returns UNVERIFIED rather than treating a REAL prediction as proof."
    )

with st.expander("System architecture"):
    st.markdown(
        "1. Text preprocessing → 2. NLP classification → 3. Claim verification → "
        "4. Evidence retrieval → 5. Attributed verdict / UNVERIFIED."
    )

with st.expander("Deployment note"):
    st.write(
        "For live Google Fact Check verification, add a Streamlit secret named "
        "`GOOGLE_FACTCHECK_API_KEY`. The Google Fact Check Tools API requires an API key."
    )

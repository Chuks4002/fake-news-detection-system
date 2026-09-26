
import streamlit as st
import joblib
import re
import math
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests

BASE = Path(__file__).resolve().parent
ARTICLE_MODEL = BASE / "fake_news_model.joblib"
HEADLINE_MODEL = BASE / "headline_model.joblib"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

st.set_page_config(
    page_title="Nigerian Fake News Detection System",
    page_icon="📰",
    layout="centered",
)

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
    vec = bundle["vectorizer"]
    clf = bundle["classifier"]
    x = vec.transform([clean(text)])
    pred = int(clf.predict(x)[0])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(x)[0]
        classes = list(getattr(clf, "classes_", range(len(probs))))
        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
        confidence = max(prob_map.values())
    else:
        score = float(clf.decision_function(x)[0])
        confidence = 1.0 / (1.0 + math.exp(-abs(score)))
        prob_map = {pred: confidence}

    label = "Potentially Fake / Misleading" if pred == 0 else "Potentially Real"
    return label, confidence, prob_map

def words(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def similarity(a, b):
    """Conservative lexical similarity for matching a returned fact-check
    to the user's actual claim."""
    aa, bb = clean(a), clean(b)
    if not aa or not bb:
        return 0.0
    seq = SequenceMatcher(None, aa, bb).ratio()
    wa, wb = words(aa), words(bb)
    if not wa or not wb:
        return seq
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
    jaccard = len(wa & wb) / max(1, len(wa | wb))
    return 0.45 * seq + 0.35 * overlap + 0.20 * jaccard

def extract_date(obj):
    # The API can expose review metadata in different nested locations.
    candidates = [
        obj.get("reviewDate"),
        obj.get("datePublished"),
        obj.get("claimReview", {}).get("datePublished")
            if isinstance(obj.get("claimReview"), dict) else None,
    ]
    for value in candidates:
        if value:
            try:
                text = str(value).replace("Z", "+00:00")
                dt = datetime.fromisoformat(text)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    return None

def time_sensitive_claim(text):
    t = text.lower()
    patterns = [
        r"\bcurrent\b", r"\bcurrently\b", r"\btoday\b", r"\bnow\b",
        r"\bthis week\b", r"\bthis month\b", r"\byesterday\b",
        r"\btomorrow\b", r"\bthis year\b", r"\blatest\b",
        r"\bpresent president\b",
    ]
    return any(re.search(p, t) for p in patterns)

def year_tokens(text):
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))

def candidate_relevance(user_claim, result):
    claim_text = result.get("claim_text", "")
    title = result.get("title", "")
    combined = f"{claim_text} {title}".strip()
    score = similarity(user_claim, combined)

    # If the user's claim names a specific year, a result about a different
    # year should not be treated as direct evidence.
    qyears = year_tokens(user_claim)
    ryears = year_tokens(combined)
    if qyears and ryears and not (qyears & ryears):
        return 0.0, "different year/context"

    # Present-time claims require a reasonably recent fact-check. Otherwise an
    # old fact-check can be incorrectly applied to a changed situation.
    if time_sensitive_claim(user_claim):
        dt = result.get("date")
        if dt is not None:
            age_days = (datetime.now(timezone.utc) - dt).days
            if age_days > 365:
                return score, "stale for a time-sensitive claim"

    return score, "matched"

def build_factcheck_queries(query):
    """Create several conservative search variants for the Google Fact Check API.

    The API search is keyword-oriented, so a single natural-language query can
    miss a published fact-check even when the fact-check claim is an excellent
    match. These variants improve retrieval without weakening the evidence
    matching performed after retrieval.
    """
    original = re.sub(r"\s+", " ", str(query)).strip()
    cleaned = clean(original)

    # Keep informative words while dropping common grammatical words. This is
    # only for retrieval; the original claim is still used for verification.
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "for", "from", "has", "have", "had", "he", "her", "his", "i",
        "if", "in", "into", "is", "it", "its", "of", "on", "or", "that",
        "the", "their", "there", "these", "they", "this", "to", "was", "were",
        "will", "with", "would", "you", "your"
    }
    informative = [w for w in cleaned.split() if w not in stopwords]

    queries = [original]

    # A compact keyword query is often more discoverable than a sentence.
    if len(informative) >= 2:
        queries.append(" ".join(informative[:12]))

    # A slightly shorter variant helps when the API ranks exact keyword
    # combinations more strongly than long natural-language sentences.
    if len(informative) > 5:
        queries.append(" ".join(informative[:8]))

    # Remove duplicates while preserving deterministic order.
    unique = []
    seen = set()
    for q in queries:
        q = q.strip()
        key = q.lower()
        if q and key not in seen:
            seen.add(key)
            unique.append(q[:500])
    return unique


def search_fact_checks(query, api_key):
    """Retrieve fact-check candidates using several query variants.

    Retrieval is intentionally broad, but a result is never accepted merely
    because it was returned by the API. candidate_relevance/select_evidence
    still decide whether it is sufficiently similar and contextually valid.
    """
    all_results = []

    for search_query in build_factcheck_queries(query):
        params = {
            "query": search_query,
            "languageCode": "en",
            "pageSize": 10,
            "key": api_key,
        }
        response = requests.get(FACTCHECK_URL, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()

        for claim in data.get("claims", []):
            claim_text = claim.get("text", "")
            for review in claim.get("claimReview", []) or []:
                publisher = review.get("publisher", {}) or {}
                rating = review.get("textualRating") or review.get("rating", {}).get("textualRating")
                url = review.get("url") or review.get("reviewUrl")
                title = review.get("title") or review.get("claimReviewed") or claim_text

                item = {
                    "claim_text": claim_text,
                    "title": title,
                    "publisher": publisher.get("name") or publisher.get("site") or "Unknown publisher",
                    "rating": rating or "Not stated",
                    "url": url,
                    "date": extract_date(review),
                    "search_query": search_query,
                }
                score, reason = candidate_relevance(query, item)
                item["score"] = score
                item["reason"] = reason
                all_results.append(item)

    # The same review can be returned for multiple query variants. Keep the
    # strongest-scoring copy so the UI remains clean and deterministic.
    deduped = {}
    for item in all_results:
        key = item.get("url") or (
            item.get("publisher", ""),
            item.get("claim_text", ""),
            item.get("rating", "")
        )
        if key not in deduped or item["score"] > deduped[key]["score"]:
            deduped[key] = item

    results = list(deduped.values())
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def select_evidence(query, results):
    if not results:
        return None

    # Conservative threshold: a related fact-check is not enough.
    best = results[0]
    if best["score"] < 0.62:
        return None

    if best["reason"] == "stale for a time-sensitive claim":
        return None

    # For short claims, require stronger lexical agreement.
    q_words = words(query)
    if len(q_words) <= 7 and best["score"] < 0.72:
        return None

    return best

# ---------------- UI ----------------

try:
    article_bundle, headline_bundle = load_models()
except Exception as exc:
    st.error("The machine-learning model could not be loaded.")
    st.exception(exc)
    st.stop()

st.title("Fake News Detection System")
st.write(
    "Natural Language Processing prototype for classifying news text and "
    "checking it against published fact-check evidence."
)

st.info(
    "The NLP model estimates linguistic classification risk. "
    "The verification result is based on matching published fact-check evidence; "
    "a missing fact-check does not mean a claim is true."
)

mode = st.radio(
    "Input type",
    ["Headline / claim", "Full article"],
    horizontal=True,
)

text = st.text_area(
    "Enter a claim or news text",
    height=220,
    placeholder="Example: The government has announced a new national holiday.",
)

if st.button("Analyse & Verify", type="primary"):
    if not text.strip():
        st.warning("Please enter text to analyse.")
        st.stop()

    min_words = 3 if mode == "Headline / claim" else 20
    if len(text.split()) < min_words:
        st.warning(
            f"Please enter at least {min_words} words for {mode.lower()} analysis."
        )
        st.stop()

    # NLP layer
    bundle = headline_bundle if mode == "Headline / claim" else article_bundle
    nlp_label, nlp_confidence, _ = model_prediction(bundle, text)

    # Fact-check layer
    api_key = st.secrets.get("GOOGLE_FACTCHECK_API_KEY", "").strip()

    evidence = None
    raw_results = []
    api_error = None

    if api_key:
        try:
            raw_results = search_fact_checks(text, api_key)
            evidence = select_evidence(text, raw_results)
        except requests.HTTPError as exc:
            api_error = f"Fact-check API returned HTTP {exc.response.status_code}."
        except requests.RequestException:
            api_error = "The fact-check service could not be reached."
        except Exception as exc:
            api_error = f"Fact-check lookup failed: {exc}"

    st.subheader("Verification result")

    if evidence:
        rating = evidence["rating"]
        rating_lower = rating.lower()

        if any(x in rating_lower for x in ["false", "fake", "incorrect", "misleading"]):
            st.error("VERDICT: Potentially Fake / Misleading")
        elif any(x in rating_lower for x in ["true", "correct", "accurate"]):
            st.success("VERDICT: Supported by Published Fact-check Evidence")
        else:
            st.warning("VERDICT: Published Fact-check Found — Review Rating")

        st.write(f"**Fact-check publisher:** {evidence['publisher']}")
        st.write(f"**Published rating:** {rating}")
        st.write(f"**Matched claim:** {evidence['claim_text']}")

        if evidence["date"]:
            st.write(
                f"**Fact-check date:** {evidence['date'].date().isoformat()}"
            )

        if evidence["url"]:
            st.link_button("Open fact-check", evidence["url"])

        st.caption(
            "The verdict above is attributed to the fact-check publisher. "
            "It is not generated by the NLP classifier. The system first checks "
            "whether the returned fact-check is sufficiently similar and "
            "contextually relevant to the submitted claim."
        )

    else:
        st.warning("VERDICT: UNVERIFIED")
        if api_error:
            st.write(api_error)
        elif not api_key:
            st.write(
                "No Google Fact Check API key is configured. The system cannot "
                "issue an evidence-based fact-check verdict."
            )
        elif raw_results:
            st.write(
                "Related fact-check records were found, but none passed the "
                "system's claim-similarity/context checks. They were not used "
                "to issue a false/true verdict."
            )
        else:
            st.write(
                "No matching published fact-check was found. This does not mean "
                "the claim is true; it means the system does not have sufficient "
                "verified evidence to issue a fake/real verdict."
            )

    st.subheader("NLP model assessment")
    if nlp_label == "Potentially Fake / Misleading":
        st.error(f"{nlp_label} — model confidence: {nlp_confidence:.1%}")
    else:
        st.success(f"{nlp_label} — model confidence: {nlp_confidence:.1%}")

    st.caption(
        "The NLP assessment is a statistical text-classification output trained "
        "on the project's labelled dataset. It is not independent fact-checking."
    )

    if raw_results and evidence is None:
        with st.expander("Why a related fact-check was not used"):
            for r in raw_results[:5]:
                date_text = r["date"].date().isoformat() if r["date"] else "date unavailable"
                st.write(
                    f"**{r['publisher']}** — {r['rating']} — similarity "
                    f"{r['score']:.2f} — {r['reason']} — {date_text}"
                )
                st.write(r["claim_text"])

with st.expander("About the system"):
    st.write(
        "The system combines an NLP classification layer with a conservative "
        "published-fact-check retrieval layer. It does not treat the absence "
        "of a fact-check as proof that a claim is true."
    )

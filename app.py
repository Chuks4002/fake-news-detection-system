import streamlit as st
import joblib
import re
import math
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent
ARTICLE_MODEL = BASE / "fake_news_model.joblib"
HEADLINE_MODEL = BASE / "headline_model.joblib"
LOCAL_FACTCHECK_CSV = BASE / "nigerian_factchecks.csv"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

st.set_page_config(page_title="Nigerian Fake News Detection System", page_icon="📰", layout="centered")

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
    return ("Potentially Fake / Misleading" if pred == 0 else "Potentially Real", confidence, prob_map)

def words(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def similarity(a, b):
    aa, bb = clean(a), clean(b)
    if not aa or not bb: return 0.0
    seq = SequenceMatcher(None, aa, bb).ratio()
    wa, wb = words(aa), words(bb)
    if not wa or not wb: return seq
    overlap = len(wa & wb) / max(1, min(len(wa), len(wb)))
    jaccard = len(wa & wb) / max(1, len(wa | wb))
    return 0.45 * seq + 0.35 * overlap + 0.20 * jaccard

def extract_date(obj):
    candidates = [obj.get("reviewDate"), obj.get("datePublished"),
                  obj.get("claimReview", {}).get("datePublished") if isinstance(obj.get("claimReview"), dict) else None]
    for value in candidates:
        if value:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception: pass
    return None

def time_sensitive_claim(text):
    return any(re.search(p, text.lower()) for p in [r"\bcurrent\b", r"\bcurrently\b", r"\btoday\b", r"\bnow\b", r"\bthis week\b", r"\bthis month\b", r"\byesterday\b", r"\btomorrow\b", r"\bthis year\b", r"\blatest\b", r"\bpresent president\b"])

def year_tokens(text):
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))

def candidate_relevance(user_claim, result):
    combined = f"{result.get('claim_text','')} {result.get('title','')}".strip()
    score = similarity(user_claim, combined)
    qyears, ryears = year_tokens(user_claim), year_tokens(combined)
    if qyears and ryears and not (qyears & ryears): return 0.0, "different year/context"
    if time_sensitive_claim(user_claim) and result.get("date") is not None:
        if (datetime.now(timezone.utc) - result["date"]).days > 365: return score, "stale for a time-sensitive claim"
    return score, "matched"

def build_factcheck_queries(query):
    original = re.sub(r"\s+", " ", str(query)).strip()
    cleaned = clean(original)
    stop = {"a","an","and","are","as","at","be","been","being","but","by","for","from","has","have","had","he","her","his","i","if","in","into","is","it","its","of","on","or","that","the","their","there","these","they","this","to","was","were","will","with","would","you","your"}
    informative = [w for w in cleaned.split() if w not in stop]
    queries = [original]
    if informative: queries.append(" ".join(informative))
    if len(informative) >= 3:
        queries += [" ".join([informative[0], informative[-1], *informative[1:-1]]), " ".join(informative[:2] + informative[-2:])]
    out=[]; seen=set()
    for q in queries:
        k=q.lower().strip()
        if k and k not in seen: seen.add(k); out.append(q[:500])
    return out

def nigeria_related(query):
    return bool(re.search(r"\b(?:nigeria|nigerian|abuja|lagos|tinubu|naira|jamb|nysc|kwara|rivers|abia)\b", clean(query)))

def _make_item(query, claim_text, title, publisher, rating, url, date, search_query="", publisher_filter=""):
    item={"claim_text":claim_text or title or "", "title":title or claim_text or "", "publisher":publisher or "Unknown publisher", "rating":rating or "Not stated", "url":url, "date":date, "search_query":search_query, "publisher_filter":publisher_filter}
    item["score"], item["reason"] = candidate_relevance(query,item)
    return item

# ---------- Nigerian local evidence corpus ----------
@st.cache_data
def load_local_factchecks():
    if not LOCAL_FACTCHECK_CSV.exists(): return []
    rows=[]
    try:
        with LOCAL_FACTCHECK_CSV.open("r",encoding="utf-8",newline="") as f:
            for r in csv.DictReader(f):
                try: r["date"]=datetime.fromisoformat(r["date"]).replace(tzinfo=timezone.utc)
                except Exception: r["date"]=None
                rows.append(r)
    except Exception: return []
    return rows

@st.cache_resource
def build_local_factcheck_index():
    rows=load_local_factchecks()
    if not rows: return None, []
    corpus=[f"{r['claim_text']} {r['title']}" for r in rows]
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=1,sublinear_tf=True,lowercase=True)
    return vec,(vec.fit_transform(corpus),rows)

def _local_factcheck_candidates(query):
    if not nigeria_related(query): return []
    vec,payload=build_local_factcheck_index()
    if vec is None: return []
    matrix,rows=payload
    scores=cosine_similarity(vec.transform([query]),matrix)[0]
    results=[]
    for score,row in zip(scores,rows):
        item=_make_item(query,row["claim_text"],row["title"],row["publisher"],row["rating"],row["url"],row["date"],"local Nigerian fact-check corpus",row["publisher"])
        item["score"]=0.65*float(score)+0.35*item["score"]
        item["reason"]="local Nigerian corpus match"
        if item["score"]>=0.42: results.append(item)
    return sorted(results,key=lambda x:x["score"],reverse=True)[:10]

def _parse_google_claims(query,data,all_results,search_query,publisher_filter=""):
    for claim in data.get("claims",[]) or []:
        for review in claim.get("claimReview",[]) or []:
            pub=review.get("publisher",{}) or {}
            all_results.append(_make_item(query,claim.get("text",""),review.get("title") or claim.get("text",""),pub.get("name") or pub.get("site"),review.get("textualRating"),review.get("url"),extract_date(review),search_query,publisher_filter))

def _search_google_once(query,api_key,publisher_filter=None,offset=0):
    params={"query":query,"languageCode":"en","pageSize":50,"offset":offset,"key":api_key}
    if publisher_filter: params["reviewPublisherSiteFilter"]=publisher_filter
    r=requests.get(FACTCHECK_URL,params=params,timeout=15); r.raise_for_status(); return r.json()

def _africacheck_candidates(query):
    if not nigeria_related(query): return []
    results=[]; search_url="https://africacheck.org/search?search_api_fulltext="+quote_plus(query)
    headers={"User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1"}
    try:
        r=requests.get(search_url,headers=headers,timeout=15); r.raise_for_status(); soup=BeautifulSoup(r.text,"html.parser")
        seen=set()
        for a in soup.find_all("a",href=True):
            href=a["href"]
            if "/fact-checks/" not in href: continue
            url="https://africacheck.org"+href if href.startswith("/") else href
            if url in seen: continue
            seen.add(url); title=" ".join(a.get_text(" ",strip=True).split())
            if len(title)<10: continue
            item=_make_item(query,title,title,"Africa Check","Not stated",url,None,query,"africacheck.org")
            if item["score"]<0.40: continue
            try:
                ar=requests.get(url,headers=headers,timeout=12)
                if ar.ok:
                    soup2=BeautifulSoup(ar.text,"html.parser")
                    for script in soup2.find_all("script",attrs={"type":"application/ld+json"}):
                        try: data=json.loads(script.string or script.get_text())
                        except Exception: continue
                        stack=list(data if isinstance(data,list) else [data]); found=False
                        while stack and not found:
                            obj=stack.pop()
                            if isinstance(obj,dict):
                                typ=obj.get("@type")
                                if typ=="ClaimReview" or (isinstance(typ,list) and "ClaimReview" in typ):
                                    rr=obj.get("reviewRating") or {}
                                    item=_make_item(query,obj.get("claimReviewed") or title,title,"Africa Check",rr.get("alternateName") or rr.get("textualRating") or "Not stated",obj.get("url") or url,extract_date(obj),query,"africacheck.org"); found=True; break
                                stack.extend(v for v in obj.values() if isinstance(v,(dict,list)))
                            elif isinstance(obj,list): stack.extend(obj)
                    if item["rating"]=="Not stated":
                        body=soup2.get_text(" ",strip=True).lower()
                        if re.search(r"\bthe claim is false\b",body): item["rating"]="False"
                        elif re.search(r"\bclaim is misleading\b",body): item["rating"]="Misleading"
            except requests.RequestException: pass
            results.append(item)
    except requests.RequestException: return []
    return sorted(results,key=lambda x:x["score"],reverse=True)[:10]

def search_fact_checks(query,api_key):
    all_results=[]
    # Local corpus is first: fast, deterministic and resilient to API/network failure.
    all_results.extend(_local_factcheck_candidates(query))
    queries=build_factcheck_queries(query)
    informative=[w for w in clean(query).split() if w not in {"a","an","and","are","as","at","be","been","being","but","by","for","from","has","have","had","he","her","his","i","if","in","into","is","it","its","of","on","or","that","the","their","there","these","they","this","to","was","were","will","with","would","you","your"}]
    if len(informative)>=3: queries.append(" ".join(informative))
    uniq=[]; seen=set()
    for q in queries:
        k=q.lower().strip()
        if k and k not in seen: seen.add(k); uniq.append(q)
    for q in uniq[:5]:
        for offset in (0,50):
            try:
                data=_search_google_once(q,api_key,offset=offset); _parse_google_claims(query,data,all_results,q)
                if not data.get("claims"): break
            except requests.RequestException: break
    if nigeria_related(query):
        for publisher in ("africacheck.org","dubawa.org"):
            for q in uniq[:3]:
                try: _parse_google_claims(query,_search_google_once(q,api_key,publisher_filter=publisher),all_results,q,publisher)
                except requests.RequestException: break
        all_results.extend(_africacheck_candidates(query))
    dedup={}
    for item in all_results:
        key=item.get("url") or (item.get("publisher",""),item.get("claim_text",""),item.get("rating",""))
        if key not in dedup or item["score"]>dedup[key]["score"]: dedup[key]=item
    return sorted(dedup.values(),key=lambda x:x["score"],reverse=True)

def select_evidence(query,results):
    if not results: return None
    best=results[0]
    # Local corpus has a slightly more tolerant threshold because its records
    # are explicitly curated published evidence; API results keep the old bar.
    threshold=0.50 if best["reason"]=="local Nigerian corpus match" else 0.62
    if best["score"]<threshold or best["reason"]=="stale for a time-sensitive claim": return None
    if len(words(query))<=7 and best["score"]<(0.62 if best["reason"]=="local Nigerian corpus match" else 0.72): return None
    return best

try:
    article_bundle,headline_bundle=load_models()
except Exception as exc:
    st.error("The machine-learning model could not be loaded."); st.exception(exc); st.stop()

st.title("Fake News Detection System")
st.write("Natural Language Processing prototype for classifying news text and checking it against published fact-check evidence.")
st.info("The NLP model estimates linguistic classification risk. The verification result is based on matching published fact-check evidence; a missing fact-check does not mean a claim is true.")
mode=st.radio("Input type",["Headline / claim","Full article"],horizontal=True)
text=st.text_area("Enter a claim or news text",height=220,placeholder="Example: The government has announced a new national holiday.")

if st.button("Analyse & Verify",type="primary"):
    if not text.strip(): st.warning("Please enter text to analyse."); st.stop()
    min_words=3 if mode=="Headline / claim" else 20
    if len(text.split())<min_words: st.warning(f"Please enter at least {min_words} words for {mode.lower()} analysis."); st.stop()
    bundle=headline_bundle if mode=="Headline / claim" else article_bundle
    nlp_label,nlp_confidence,_=model_prediction(bundle,text)
    api_key=st.secrets.get("GOOGLE_FACTCHECK_API_KEY","").strip()
    evidence=None; raw_results=[]; api_error=None
    if api_key:
        try:
            raw_results=search_fact_checks(text,api_key); evidence=select_evidence(text,raw_results)
        except requests.HTTPError as exc: api_error=f"Fact-check API returned HTTP {exc.response.status_code}."
        except requests.RequestException: api_error="The fact-check service could not be reached."
        except Exception as exc: api_error=f"Fact-check lookup failed: {exc}"
    st.subheader("Verification result")
    if evidence:
        rating=evidence["rating"]; rl=rating.lower()
        if any(x in rl for x in ["false","fake","incorrect","misleading"]): st.error("VERDICT: Potentially Fake / Misleading")
        elif any(x in rl for x in ["true","correct","accurate"]): st.success("VERDICT: Supported by Published Fact-check Evidence")
        else: st.warning("VERDICT: Published Fact-check Found — Review Rating")
        st.write(f"**Fact-check publisher:** {evidence['publisher']}")
        st.write(f"**Published rating:** {rating}")
        st.write(f"**Matched claim:** {evidence['claim_text']}")
        if evidence["date"]: st.write(f"**Fact-check date:** {evidence['date'].date().isoformat()}")
        if evidence["url"]: st.link_button("Open fact-check",evidence["url"])
        if evidence.get("search_query")=="local Nigerian fact-check corpus": st.info("This record was matched from the system's local Nigerian fact-check evidence corpus. The source remains the named fact-check publisher.")
        st.caption("The verdict above is attributed to the fact-check publisher. It is not generated by the NLP classifier. The system first checks whether the returned fact-check is sufficiently similar and contextually relevant to the submitted claim.")
    else:
        st.warning("VERDICT: UNVERIFIED")
        if api_error: st.write(api_error)
        elif not api_key: st.write("No Google Fact Check API key is configured. The system cannot issue an evidence-based fact-check verdict.")
        elif raw_results: st.write("Related fact-check records were found, but none passed the system's claim-similarity/context checks. They were not used to issue a false/true verdict.")
        else: st.write("No matching published fact-check was found. This does not mean the claim is true; it means the system does not have sufficient verified evidence to issue a fake/real verdict.")
    st.subheader("NLP model assessment")
    if nlp_label=="Potentially Fake / Misleading": st.error(f"{nlp_label} — model confidence: {nlp_confidence:.1%}")
    else: st.success(f"{nlp_label} — model confidence: {nlp_confidence:.1%}")
    st.caption("The NLP assessment is a statistical text-classification output trained on the project's labelled dataset. It is not independent fact-checking.")
    if raw_results and evidence is None:
        with st.expander("Why a related fact-check was not used"):
            for r in raw_results[:5]:
                d=r["date"].date().isoformat() if r["date"] else "date unavailable"
                st.write(f"**{r['publisher']}** — {r['rating']} — similarity {r['score']:.2f} — {r['reason']} — {d}")
                st.write(r["claim_text"])

with st.expander("About the system"):
    st.write("The system combines an NLP classification layer with a conservative published-fact-check retrieval layer. It does not treat the absence of a fact-check as proof that a claim is true.")

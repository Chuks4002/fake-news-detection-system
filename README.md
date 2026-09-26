# Fake News Detection System — Implementation Package

## Current verified experiment
Dataset: WELFake_Dataset.csv
Preprocessing: missing text removed; duplicate combined title/body records removed.
Final corpus: 63,676 unique articles.
Split: 80% training / 20% held-out testing, stratified, random_state=42.
Representation: HashingVectorizer, 2^18 features, word unigrams.
Classifier: SGDClassifier with logistic loss.
Evaluation: accuracy, macro precision, macro recall, macro F1, confusion matrix.

## Current results
Accuracy: 94.17%
Macro precision: 94.16%
Macro recall: 94.08%
Macro F1: 94.12%

## Functional testing
20 real held-out articles were passed through the same prediction pipeline.
19/20 were classified correctly (95% sample pass rate).
The single failure is retained in the test CSV rather than removed.

## Run
Install requirements:
pip install -r requirements.txt

Start the interface:
streamlit run app.py

Important:
The WELFake corpus is not Nigerian social-media data. It is the primary fake/real classification corpus. Nigerian social-media relevance must be handled as a contextual limitation/evaluation discussion unless a separately labelled Nigerian fake-news corpus is obtained.

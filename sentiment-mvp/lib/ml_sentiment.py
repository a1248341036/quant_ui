# -*- coding: utf-8 -*-
"""中文金融情绪 ML 分类器。

训练数据来自开源项目 algosenses/Stock_Market_Sentiment_Analysis：
东财股吧正/负股评各 4607 条（2017-04 ~ 2018-05）。用 char n-gram TF-IDF + LR
训练二分类，输出正/负概率；中性用置信度阈值圈出来。模型落盘到
项目内 models/，默认评分方法仍是词典，可用 method=ml 切换。
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = _PROJECT_ROOT / "datasets" / "algosenses"
MODEL_DIR = _PROJECT_ROOT / "models"
POS_FILE = DATASET_DIR / "positive.txt"
NEG_FILE = DATASET_DIR / "negative.txt"
MODEL_FILE = MODEL_DIR / "ml_model.joblib"
VEC_FILE = MODEL_DIR / "ml_vectorizer.joblib"
META_FILE = MODEL_DIR / "ml_metrics.json"

# 中性判定：prob(positive) 落在 [1-pos_thr, pos_thr] 之外才判正/负
POS_THRESHOLD = 0.60
NEG_THRESHOLD = 0.40


def ensure_data() -> None:
    """数据缺失时从 GitHub 拉取（algosenses/Stock_Market_Sentiment_Analysis）。"""
    if POS_FILE.exists() and NEG_FILE.exists():
        return
    import urllib.request
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/algosenses/Stock_Market_Sentiment_Analysis/master/data"
    for name in ("positive.txt", "negative.txt"):
        target = DATASET_DIR / name
        if not target.exists():
            print(f"download {name} ...")
            urllib.request.urlretrieve(f"{base}/{name}", target)


def load_labeled() -> pd.DataFrame:
    ensure_data()
    pos = POS_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    neg = NEG_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    pos = [s.strip() for s in pos if s.strip()]
    neg = [s.strip() for s in neg if s.strip()]
    df = pd.DataFrame({"text": pos + neg, "label": ["positive"] * len(pos) + ["negative"] * len(neg)})
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df


def train(force: bool = False) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists() and VEC_FILE.exists() and not force:
        return json.loads(META_FILE.read_text(encoding="utf-8"))

    df = load_labeled()
    print(f"labeled samples: {len(df)} (pos {int((df['label']=='positive').sum())} / neg {int((df['label']=='negative').sum())})")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_df=0.95,
        sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, pos_label="positive")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_threshold": POS_THRESHOLD,
        "neg_threshold": NEG_THRESHOLD,
        "report": classification_report(y_test, pred, output_dict=True),
    }
    print(classification_report(y_test, pred))
    print(f"train acc={metrics['accuracy']:.4f} f1(pos)={metrics['f1']:.4f}")

    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vec, VEC_FILE)
    META_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {MODEL_FILE}")
    return metrics


_model = None
_vec = None


def _load():
    global _model, _vec
    if _model is None and MODEL_FILE.exists():
        _model = joblib.load(MODEL_FILE)
        _vec = joblib.load(VEC_FILE)
    return _model, _vec


def score_text(title, content=""):
    """返回 (label, score)。score ∈ [-1,1]，正=看多。无模型时返回 None。"""
    clf, vec = _load()
    if clf is None:
        return None
    text = f"{title} {content}"[:800]
    x = vec.transform([text])
    proba = clf.predict_proba(x)[0]
    idx = {c: i for i, c in enumerate(clf.classes_)}
    p_pos = float(proba[idx.get("positive", 0)])
    if p_pos >= POS_THRESHOLD:
        label = "positive"
    elif p_pos <= NEG_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"
    score = round(2.0 * p_pos - 1.0, 4)
    return label, score


def model_ready() -> bool:
    return MODEL_FILE.exists() and VEC_FILE.exists()

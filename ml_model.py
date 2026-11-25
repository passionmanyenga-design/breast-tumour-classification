from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             confusion_matrix)


def train_and_evaluate(X_tr, X_te, y_tr, y_te):
    """Fit scaled logistic regression on the training split and score it on
    the held-out test split.

    Returns dict with 'accuracy', 'recall' (malignant=1), 'auc', 'fn', 'fp'.
    """
    # The pipeline fits the scaler on training data only (no leakage).
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return {
        'accuracy': accuracy_score(y_te, pred),
        'recall': recall_score(y_te, pred),
        'auc': roc_auc_score(y_te, proba),
        'fn': int(fn),   # malignant tumours missed
        'fp': int(fp),   # benign tumours flagged as malignant
    }

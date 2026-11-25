"""Repeat the whole pipeline over several train/test splits, and compare the
searched subset with random subsets of the same size. Results -> results.json"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from main import run
from data_processing import load_data
from ml_model import train_and_evaluate

SEEDS = list(range(8))
rows = []
for s in SEEDS:
    r = run(seed=s, verbose=False)
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=s)
    k = len(r['subset'])
    rng = np.random.default_rng(s)
    rand_acc, rand_rec = [], []
    for _ in range(30):
        idx = sorted(rng.choice(X.shape[1], size=k, replace=False))
        m = train_and_evaluate(X_tr.iloc[:, idx], X_te.iloc[:, idx], y_tr, y_te)
        rand_acc.append(m['accuracy']); rand_rec.append(m['recall'])
    rows.append(dict(seed=s, k=k, n_evals=r['n_evals'], cv=r['cv_score'],
                     search_time=r['search_time'],
                     full=r['full'], sub=r['sub'],
                     rand_acc=float(np.mean(rand_acc)),
                     rand_rec=float(np.mean(rand_rec)),
                     subset=[X.columns[i] for i in r['subset']]))
    print(s, k, r['n_evals'], round(r['search_time'], 1),
          round(r['full']['accuracy'], 3), round(r['sub']['accuracy'], 3), flush=True)

json.dump(rows, open('results.json', 'w'), indent=1)
print('done')

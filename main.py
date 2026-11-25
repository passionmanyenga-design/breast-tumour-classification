import time
from sklearn.model_selection import train_test_split
from data_processing import load_data
from search_algorithm import hill_climb
from ml_model import train_and_evaluate


def run(seed=42, verbose=True):
    X, y = load_data()
    # ONE split up front. The search sees only the training part, so the
    # test set stays completely unseen by both the search and the model.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    t0 = time.time()
    subset, cv_score, n_evals = hill_climb(X_tr, y_tr, verbose=verbose)
    search_time = time.time() - t0

    t0 = time.time()
    full = train_and_evaluate(X_tr, X_te, y_tr, y_te)
    full_time = time.time() - t0

    t0 = time.time()
    sub = train_and_evaluate(X_tr.iloc[:, subset], X_te.iloc[:, subset],
                             y_tr, y_te)
    sub_time = time.time() - t0

    return dict(X=X, subset=subset, cv_score=cv_score, n_evals=n_evals,
                search_time=search_time, full=full, full_time=full_time,
                sub=sub, sub_time=sub_time)


def main():
    print('=== Search: hill-climbing feature selection (training split only) ===')
    r = run()
    X, subset = r['X'], r['subset']
    full, sub = r['full'], r['sub']
    print(f'Dataset: {X.shape[0]} patients, {X.shape[1]} features')
    print(f'  -> {len(subset)} features, CV acc={r["cv_score"]:.4f}, '
          f'{r["n_evals"]} subsets evaluated, time={r["search_time"]:.1f}s\n')

    print('=' * 70)
    print(f'{"Method":<30}{"Acc":>7}{"Recall":>8}{"AUC":>7}{"FN":>4}{"FP":>4}{"Time(s)":>9}')
    print('-' * 70)
    print(f'{"LR, all 30 features":<30}{full["accuracy"]:>7.3f}{full["recall"]:>8.3f}'
          f'{full["auc"]:>7.3f}{full["fn"]:>4}{full["fp"]:>4}{r["full_time"]:>9.3f}')
    print(f'{"LR, searched subset":<30}{sub["accuracy"]:>7.3f}{sub["recall"]:>8.3f}'
          f'{sub["auc"]:>7.3f}{sub["fn"]:>4}{sub["fp"]:>4}{r["sub_time"]:>9.3f}')
    print(f'{"Hill-climb search (CV acc)":<30}{r["cv_score"]:>7.3f}'
          f'{"-":>8}{"-":>7}{"-":>4}{"-":>4}{r["search_time"]:>9.1f}')
    print('=' * 70)
    print(f'\nSelected features ({len(subset)}): '
          f'{[X.columns[i] for i in subset]}')


if __name__ == '__main__':
    main()

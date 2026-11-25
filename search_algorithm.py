from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def evaluate_subset(X, y, subset, cv=5):
    """Mean CV accuracy of a random forest on the given column indices.
    An empty subset scores 0 so the search can never accept it."""
    if len(subset) == 0:
        return 0.0
    model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    # Fixed, shuffled folds: every candidate subset is scored on identical
    # splits, so score differences come from the features, not the folds.
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
    scores = cross_val_score(model, X.iloc[:, subset], y, cv=folds)
    return scores.mean()


def hill_climb(X, y, cv=5, verbose=True):
    """Greedy backward elimination (first-improvement hill climbing).

    State: a subset of feature indices. Neighbours: the subset with one
    feature removed. A removal is accepted if CV accuracy does not drop,
    then the scan restarts. Stops when a full pass accepts nothing
    (local optimum).

    Returns: (subset, best_score, n_evaluations)
    """
    current = list(range(X.shape[1]))
    best_score = evaluate_subset(X, y, current, cv)
    n_evals = 1

    if verbose:
        print(f'  start: {len(current)} features, CV acc={best_score:.4f}')

    improved = True
    while improved:
        improved = False
        for i in list(current):
            candidate = [j for j in current if j != i]
            score = evaluate_subset(X, y, candidate, cv)
            n_evals += 1
            if score >= best_score:
                dropped = X.columns[i]
                current, best_score, improved = candidate, score, True
                if verbose:
                    print(f'  drop {dropped:>25} -> {len(current)} features, '
                          f'CV acc={best_score:.4f}')
                break  # restart the scan from the smaller subset

    # NB: return sits OUTSIDE the while loop (the original returned inside
    # it, so the search stopped after the first accepted removal).
    return current, best_score, n_evals

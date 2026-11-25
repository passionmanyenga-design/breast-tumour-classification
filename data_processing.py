import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
    'concavity', 'concave_points', 'symmetry', 'fractal_dimension'
]


def load_data(path='data/wdbc.data'):
    df = pd.read_csv(path, header=None)
    column_names = (
        ['ID', 'Diagnosis']
        + [f'{f}_mean' for f in FEATURES]
        + [f'{f}_se' for f in FEATURES]
        + [f'{f}_worst' for f in FEATURES]
    )
    df.columns = column_names
    df = df.drop('ID', axis=1)
    X = df.drop('Diagnosis', axis=1)
    y = LabelEncoder().fit_transform(df['Diagnosis'])
    return X, y

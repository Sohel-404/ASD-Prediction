import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer


def loadData(path):

    df = pd.read_csv(path)
    features = df.drop(columns=['Sample','Condition'])
    target = df['Condition'].map({"Control":0, "ASD":1})

    print("Number of genes: ", features.shape[1])

    X = features
    y = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=12
    )

    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(100, X.shape[0]),
        random_state=12
    )

    qt.fit(X_train)
    X_train_q = pd.DataFrame(
        qt.transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )

    X_test_q = pd.DataFrame(
        qt.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    scaler = StandardScaler()

    scaler.fit(X_train_q)
    X_train_s = pd.DataFrame(
        scaler.transform(X_train_q),
        index=X_train_q.index,
        columns=X_train_q.columns
    )

    X_test_s = pd.DataFrame(
        scaler.transform(X_test_q),
        index=X_test_q.index,
        columns=X_test_q.columns
    )

    return df, X_train_s, X_test_s, y_train, y_test
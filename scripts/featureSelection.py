import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def varFilter(X_train, X_test, threshold=0.7):

    vt = VarianceThreshold(threshold)
    vt.fit(X_train)
    X_train_vt = vt.transform(X_train)
    X_test_vt = vt.transform(X_test)

    selected_features = X_train.columns[vt.get_support()]

    X_train_vt = pd.DataFrame(X_train_vt, index=X_train.index, columns=selected_features)
    X_test_vt = pd.DataFrame(X_test_vt, index=X_test.index, columns=selected_features)

    print(f"[VT] Features after variance filtering: {X_train_vt.shape[1]}")

    return X_train_vt, X_test_vt, selected_features, vt



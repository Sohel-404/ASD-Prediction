#Importing the modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier
import os
import random

SEED = 12
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["HYPEROPT_FMIN_SEED"] = str(SEED)
os.environ["XGBOOST_RANDOM_STATE"] = str(SEED)

#Loading the data and Normalization
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = df.drop(columns=["Sample", "Condition"])
    print("No of features: ", features.shape[1])

    # Quantile normalization
    features_qn = pd.DataFrame(
        quantile_transform(features, axis=0, n_quantiles=100,
                           output_distribution='normal', copy=True),
        index=features.index, columns=features.columns
    )

    # Z-score normalization
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_qn),
        index=features_qn.index, columns=features_qn.columns
    )
    X = features_scaled
    y = df["Condition"].map({"ASD": 1, "Control": 0})
    return X, y, df

#For SHAP Analysis
def run_shap_analysis(model, X_train_res, X_test, y_test):
    """
    Runs SHAP analysis on a trained model and generates plots.
    """

    explainer = shap.TreeExplainer(model, X_train_res.sample(50, random_state=42))
    shap_values = explainer.shap_values(X_test)

    # --- SHAP summary plots ---
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)
    shap.summary_plot(shap_values, X_test, show=True)

    # --- Waterfall plot (Control) ---
    neg_pos = np.where(y_test.values == 0)[0][0]
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[neg_pos],
            base_values=explainer.expected_value,
            data=X_test.iloc[neg_pos],
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()

    # --- Waterfall plot (ASD) ---
    pos_pos = np.where(y_test.values == 1)[0][0]
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[pos_pos],
            base_values=explainer.expected_value,
            data=X_test.iloc[pos_pos],
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


    # --- Violin plot ---
    shap.plots.violin(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test,
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()


    # --- Feature importance bar ---
    shap.plots.bar(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=X_test,
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    # Fixing of margins
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.subplots_adjust(left=0.25)
    plt.show()



def main():
    # Step 1: Load dataset
    file_path = "data/ML_dataset.csv"
    X, y, df = load_data(file_path)

    # Step 2: Train-test split (Train: 70% and Test: 30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    # Step 3: Variance threshold for Feature Selection
    vt = VarianceThreshold(threshold=0.7) #removes features whose variance is ≤ 0.7
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)
    print(f"Features after VarianceThreshold: {X_train_vt.shape[1]}")
    selected_vt_features = X_train.columns[vt.get_support()]

    # Step 4: Handles Class imbalance
    smote = SMOTE(sampling_strategy={0: 51, 1: 63}, k_neighbors=8, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_vt, y_train)
    print("Before SMOTE:", y_train.value_counts())
    print("After SMOTE:", y_train_res.value_counts())

    # Step 5: Feature selection to get best features
    selector = SelectKBest(score_func=mutual_info_classif, k=80)
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test_vt)

    selected_features = selected_vt_features[selector.get_support()]
    print(f"Number of features after selection: {X_train_selected.shape[1]}")

    # Step 6: Hyperopt space
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
    }

    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])

        clf = XGBClassifier(**params, random_state=42, eval_metric="logloss")
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in folds.split(X_train_selected, y_train_res):
            x_tr, x_val_fold = X_train_selected[train_idx], X_train_selected[val_idx]
            y_tr, y_val_fold = y_train_res[train_idx], y_train_res[val_idx]

            clf.fit(x_tr, y_tr)
            y_pred = clf.predict(x_val_fold)
            f1_scores.append(f1_score(y_val_fold, y_pred, average='macro'))
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)

        # Print for tracking the parameters
        print(f"Params: {params} | Mean F1: {mean_f1:.4f} | Std F1: {std_f1:.4f}")

        return {
            'loss': -mean_f1,
            'status': STATUS_OK,
            'params': params,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        }


    # Step 7: Hyperopt (optimization)
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest,
                max_evals=30, trials=trials, rstate=np.random.default_rng(42))
    best_trial = sorted(trials.results, key=lambda x: x['loss'])[0]
    best_params = best_trial['params']

    # Step 8: Final model (XGBoost model using all the optimal parameters from Hyperopt)
    clf = XGBClassifier(**best_params, random_state=42, eval_metric="logloss")
    clf.fit(X_train_selected, y_train_res)

    # Step 9: Evaluation of model performance
    print("\nBest XGBoost Params:", best_params)
    print(f"Cross-val Mean F1: {best_trial['mean_f1']:.4f}")
    print(f"Cross-val Std F1: {best_trial['std_f1']:.4f}")
    y_pred_train=clf.predict(X_train_selected)
    y_pred_test=clf.predict(X_test_selected)
    print(f"Train Accuracy: {accuracy_score(y_train_res,y_pred_train )*100: .2f}%")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)*100: .2f}%")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, clf.predict(X_test_selected), target_names=["Control", "ASD"]))
    print(confusion_matrix(y_test, clf.predict(X_test_selected)))

    # --- ROC-AUC ---
    y_prob = clf.predict_proba(X_test_selected)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC score: {roc_auc:.2f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Final XGBoost Model")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # --- Run SHAP analysis ---
    run_shap_analysis(clf, pd.DataFrame(X_train_selected, columns=selected_features),
                      pd.DataFrame(X_test_selected, columns=selected_features), y_test)

    # --- Final single test sample prediction ---
    print("\n### Single Test Sample Prediction ###")

    # Test sample choice
    sample_idx = 5

    # To make sure that the index is valid
    if sample_idx < len(X_test_selected):
        X_single = X_test_selected[sample_idx].reshape(1, -1)
        y_single_true = y_test.iloc[sample_idx]

        # Using the trained final model
        y_single_pred = clf.predict(X_single)[0]
        y_single_prob = clf.predict_proba(X_single)[0, 1]

        print(f"Test Sample Index: {sample_idx}")
        print(f"True Label: {y_single_true} ({'ASD' if y_single_true == 1 else 'Control'})")
        print(f"Predicted Label: {y_single_pred} ({'ASD' if y_single_pred == 1 else 'Control'})")
        print(f"Predicted Probability (ASD): {y_single_prob:.4f}")
    else:
        print(f"Invalid sample_idx {sample_idx}. Maximum available is {len(X_test_selected) - 1}.")


if __name__ == "__main__":
    main()


#--------Results interpreted as-----------
#Best XGBoost Params: {'colsample_bytree': 0.6817467614858493, 'gamma': 0.10741104861225978, 'learning_rate': 0.05020810119372847, 'max_depth': 8, 'n_estimators': 160, 'subsample': 0.9309660772445034}
# Cross-val Mean F1: 0.8681
# Cross-val Std F1: 0.1194
# Train Accuracy:  100.00%
# Test Accuracy:  86.67%

# Classification Report (Test):
#               precision    recall  f1-score   support

#      Control       0.92      0.71      0.80        17
#          ASD       0.84      0.96      0.90        28

#     accuracy                           0.87        45
#    macro avg       0.88      0.84      0.85        45
# weighted avg       0.87      0.87      0.86        45

# [[12  5]
#  [ 1 27]]
# ROC-AUC score: 0.83

 ### Single Test Sample Prediction ###
# Test Sample Index: 5
# True Label: 1 (ASD)
# Predicted Label: 1 (ASD)
# Predicted Probability (ASD): 0.9383
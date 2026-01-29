import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier

SEED = 42

CONFIG = {
    'test_size': 0.3,
    'variance_threshold': 0.1,
    'n_features': 80,
    'cv_folds': 5,
    'hyperopt_evals': 30,
    'smote_k_neighbors': 5
}


def setSeed(seed=SEED):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.default_rng(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Data Loading 
def loadData(file_path):
    """
    Load dataset
    
    Args:
        file_path: Path to csv file
    
    Returns:
        X: Normalized features (genes)
        y: Binary lables (ASD=1, Control=0)
    """    
    print("Loading data...")
    
    df = pd.read_csv(file_path)
    
    # Extract features and labels
    features = df.drop(columns=["Sample", "Condition"])
    feature_names = features.columns.tolist()
    
    print(f"Loaded {features.shape[1]} genes from {len(df)} samples.")
    print(f"Class Distribution: {df["Condition"].value_counts().to_dict()}")
    
    # Check for missing values
    if features.isnull().any().any():
        print("Warning: Missing values detected. Imputing with median...")
        features = features.fillna(features.median())
        
    # Encode labels
    y = df["Condition"].map({"ASD":1, "Control": 0})
    
    if y.isnull().any():
        raise ValueError("Invalid lables found. Check 'Condition' column.")
    
    return features, y, feature_names

def normalizeData(X_train, X_test):
    """
    Apply normlization after spliting data
    
    Args:
        :param X_train: Training features
        :param X_test: Test features
    
    Returns:
        X_train_norm, X_test_norm: Standardized features
    """
    print("Normalizing Data...")
    print("Method: StandardScaler (z-score normalization)")
    
    scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_test_norm
    

def selectFeatures(X_train, y_train, X_test, feature_names, config):
    """
    Select Features

    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        feature_names: Gene names
        config: Configuration dictionary

    Returns:
        X_train_selected, X_test_selected: Filtered features
        selected_features: Names of selected genes 
    """
    print("Feature Selection...")
    
    # Step 1: Remove low variance features
    print(f"Step 1: variance Threshold {config['variance_threshold']}")
    vt = VarianceThreshold(threshold=config['variance_threshold'])
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)
    
    vt_features = np.array(feature_names)[vt.get_support()].tolist()
    print(f"Retained {len(vt_features)} genes after variance filter")
    
    # Stage 2: Select top K features 
    print(f"Step 2: Mutual Information (top{config['n_features']})")
    
    k = min(config['n_features'], X_train_vt.shape[1])
    if k < config['n_features']:
        print(f"Adjusting K to {k}")
        
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_vt, y_train)
    X_test_selected = selector.transform(X_test_vt)
    
    selected_features = np.array(vt_features)[selector.get_support()].tolist()
    print(f"Final feature count: {len(selected_features)}")
    
    # Feature scores 
    feature_scores = pd.DataFrame({
        'Gene': selected_features,
        'MI_Score': selector.scores_[selector.get_support()]
    }).sort_values('MI_Score', ascending=False)
    
    print(f"\n Top 5 genes ny MI score:")
    print(feature_scores.head().to_string(index=False))
    
    return X_train_selected, X_test_selected, selected_features


def handleImbalance(X_train, y_train, config):
    """
    Apply SMOTE

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary

    Returns:
        X_resampled, y_resampled: Balanced training
    """
    print("Handling class imbalance...")
    
    class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"Before SMOTE: {class_counts.to_dict()}")
    
    minority_class_size = class_counts.min()
    
    # Adaptive K-neighbors
    k_neighbors = min(config['smote_k_neighbors'], minority_class_size - 1)
    
    if k_neighbors < 1:
        print("Warning: Too few minority samples for SMOTE. Using original data.")
        return X_train, y_train
    
    # Auto stratergy: balance to equal class sizes
    smote = SMOTE(
        sampling_strategy='auto',
        k_neighbors=k_neighbors,
        random_state=SEED
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    print(f"After SMOTE: {resampled_counts.to_dict()}")
    print(f"Generated {len(y_resampled) - len(y_train)} synthetic samples")
    
    return X_resampled, y_resampled


def optimizeHyperparameters(X_train, y_train, config):
    """
    Implement Hyperopt with cross-validation for a subset of dataset.

    Args:
        X_train: Training_features
        y_train: Training labels
        config: Configuration dictionary

    Returns:
        best_params: Optimal hyperparameters
        best_score: Best CV score schieved
    """
    print("Optimizing hyperparameters...")
    print(f"    Using {config['cv_folds']}-fold strtified CV")
    print(f"    Max evaluations: {config['hyperopt_evals']}")
    
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'gamma': hp.uniform('gamma', 0, 5),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.01), np.log(10))
    }
    
    best_scores = {'score': -np.inf}
    
    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Stratified K-fold
        skf = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=SEED)
        
        f1_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            clf = XGBClassifier(
                **params,
                random_state=SEED,
                eval_metric="logloss",
                use_label_encoder=False
            )
            
            clf.fit(X_tr, y_tr, verbose=False)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            f1_scores.append(f1)
            
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        # Update best scores
        if mean_f1 > best_scores['score']:
            best_scores['score'] = mean_f1
            
        print(f"    Trial | Mean F1: {mean_f1:.4f} +/- {std_f1:.4f} ")

        return {
            'loss': -mean_f1,
            'status': STATUS_OK,
            'params': params,
            'mean_f1': mean_f1,
            'std_f1': std_f1
        }
        
    # Run optimization
    
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=config['hyperopt_evals'],
        trials=trials,
        rstate=np.random.default_rng(SEED),
        verbose=False
    )
    
    # Get best trial
    best_trial = min(trials.results, key=lambda x:x['loss'])
    best_params = best_trial['params']
    
    print("Optimization Complete")
    print(f"    Best CV F1: {best_trial['mean_f1']:.4f} +/- {best_trial['std_f1']:.4f}")
    print(f"    Best params: {best_params}")
    
    return best_params, best_trial['mean_f1']


def modelTraining(best_params, X_train, X_test, y_train, y_test):
    print("Training model...")
    
    clf = XGBClassifier(
        **best_params,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    clf.fit(X_train, y_train, verbose=False)
    print("    Model training complete.")
    
    # Prediction
    print("Model Evaluation...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='macro')
    test_auc = roc_auc_score(y_test, y_prob_test)
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_auc': test_auc
    }
    
    # Results
    print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    
    print("Classification Report:")
    clf_report = classification_report(y_test, y_pred_test, target_names=["Control", "ASD"])
    print(clf_report)
    
    print("Confusion Matrix: ")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Plot ROC curve
    plot(y_test, y_prob_test, test_auc)
    
    return clf, metrics
    
def plot(y_test, y_prob, auc_score):
    """
    Generate and display ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, 
             label="ROC curve")
    plt.plot([0,1], [0,1], color="red", lw=2, 
             linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost Classifier")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    

def shapAnalysis(model, X_train, X_test, y_test, feature_names):
    
    print("\nSHAP Analysis...")
    
    # Convert to Dataframes with gene names
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    background = X_train_df.sample(n=min(50, len(X_train_df)), random_state=SEED)
    
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test_df)
    
    # Summary plot 
    print("    Feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.tight_layout()
    plt.show()
    


def main():
    # Setup
    setSeed(SEED)

    # 1: Load Data
    file_path = "data/ML_dataset.csv"
    X, y, feature_names = loadData(file_path)

    # 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG['test_size'], stratify=y, random_state=SEED
    )
    print(f"\nSplit: {len(X_train)} train | {len(X_test)} test samples")

    # 3: Normalize data
    X_train_norm, X_test_norm = normalizeData(X_train, X_test)
    
    # 4: Feature selection
    X_train_sel, X_test_sel, selected_features = selectFeatures(
        X_train_norm, y_train, X_test_norm, feature_names, CONFIG
    )
    
    # 5: Handle class imbalance (SMOTE)
    X_train_bal, y_train_bal = handleImbalance(
        X_train_sel, y_train, CONFIG
    )

    # 6: Hyperparameter optimization
    best_params, best_cv_score = optimizeHyperparameters(X_train_bal, y_train_bal, CONFIG)
    
    # 7: Train model
    clf, metrics = modelTraining(
        best_params, X_train_bal, X_test_sel, y_train_bal, y_test
    )
    
    # 8: SHAP interpretability analysis
    shapAnalysis(clf, X_train_bal, X_test_sel, y_test, selected_features)
       
    
if __name__=="__main__":
    main()
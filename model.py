import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, roc_auc_score, 
                             roc_curve)
from sklearn.preprocessing import StandardScaler, quantile_transform
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from xgboost import XGBClassifier


# Configuration

SEED = 12
CONFIG = {
    'test_size': 0.3,
    'variance_threshold': 0.1,      # Lower threshold for normalized data
    'n_features': 80,
    'cv_folds': 5,                  # 5-fold is more standard for small datasets
    'hyperopt_evals': 30,
    'smote_k_neighbors': 5          # Adaptive based on minority class
}

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Data Loading and Preprocessing

def load_and_preprocess(file_path):
    """
    Load dataset and apply proper normalization for RNA-seq data.
    
    RNA-seq specific considerations:
    - Quantile normalization handles between-sample differences
    - Z-score handles feature scaling
    - Order matters: quantile first to handle distributional differences
    
    Args:
        file_path: Path to ML-ready CSV file
        
    Returns:
        X: Normalized features (genes)
        y: Binary labels (ASD=1, Control=0)
        feature_names: Gene names
    """
    print("\nLoading and Preprocessing Data...")
    df = pd.read_csv(file_path)
    
    # Extract features and labels
    features = df.drop(columns=["Sample", "Condition"])
    feature_names = features.columns.tolist()
    
    print(f"Loaded {features.shape[1]} genes from {len(df)} samples")
    print(f"Class distribution: {df['Condition'].value_counts().to_dict()}")
    
    # Check for missing values
    if features.isnull().any().any():
        print("Warning: Missing values detected. Imputing with median...")
        features = features.fillna(features.median())
    
    # Quantile normalization
    print("Applying quantile normalization...")
    features_qn = pd.DataFrame(
        quantile_transform(features, axis=0, n_quantiles=min(100, len(features)),
                          output_distribution='normal', copy=True),
        index=features.index, columns=features.columns
    )
    
    # Z-score normalization (feature scaling)
    print("Applying z-score normalization...")
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features_qn),
        index=features_qn.index, columns=features_qn.columns
    )
    
    # Encode labels
    y = df["Condition"].map({"ASD": 1, "Control": 0})
    
    if y.isnull().any():
        raise ValueError("Invalid labels found. Check 'Condition' column.")
    
    return features_scaled, y, feature_names


# Feature Selection

def select_features(X_train, y_train, X_test, feature_names, config):
    
    """
    Proper feature selection pipeline for RNA-seq data.
    
    Args:
        X_train, y_train: Training data
        X_test: Test data
        feature_names: Original gene names
        config: Configuration dictionary
        
    Returns:
        X_train_selected, X_test_selected: Filtered features
        selected_features: Names of selected genes
    """
    print("\nFeature Selection...")
    
    # Stage 1: Remove low-variance features
    # For normalized data, variance threshold should be lower
    print(f"Stage 1: Variance Threshold ({config['variance_threshold']})")
    vt = VarianceThreshold(threshold=config['variance_threshold'])
    X_train_vt = vt.fit_transform(X_train)
    X_test_vt = vt.transform(X_test)
    
    features_after_vt = np.array(feature_names)[vt.get_support()].tolist()
    print(f"Retained {len(features_after_vt)} genes after variance filter")
    
    # Stage 2: Select top K features using mutual information
    # MI is robust for gene expression data
    print(f"Stage 2: Mutual Information (top {config['n_features']})")
    
    # Adjust K if needed
    k = min(config['n_features'], X_train_vt.shape[1])
    if k < config['n_features']:
        print(f"Adjusting K to {k} (limited by available features)")
    
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_vt, y_train)
    X_test_selected = selector.transform(X_test_vt)
    
    selected_features = np.array(features_after_vt)[selector.get_support()].tolist()
    print(f"Final feature count: {len(selected_features)}")
    
    # Get feature scores for inspection
    feature_scores = pd.DataFrame({
        'Gene': selected_features,
        'MI_Score': selector.scores_[selector.get_support()]
    }).sort_values('MI_Score', ascending=False)
    
    print(f"\n  Top 5 genes by MI score:")
    print(feature_scores.head().to_string(index=False))
    
    return X_train_selected, X_test_selected, selected_features


# Class Imbalance Handling

def handle_imbalance(X_train, y_train, config):
    """
    Apply SMOTE AFTER feature selection to prevent data leakage.
    
    CRITICAL FIX: SMOTE should be applied:
    1. AFTER feature selection (not before)
    2. Only on training data
    3. With adaptive k_neighbors based on minority class size
    
    Args:
        X_train: Training features (already selected)
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        X_resampled, y_resampled: Balanced training data
    """
    print("\n Handling Class Imbalance ")
    
    class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"Before SMOTE: {class_counts.to_dict()}")
    
    minority_class_size = class_counts.min()
    
    # Adaptive k_neighbors: must be less than minority class size
    k_neighbors = min(config['smote_k_neighbors'], minority_class_size - 1)
    
    if k_neighbors < 1:
        print("   Warning: Too few minority samples for SMOTE. Using original data.")
        return X_train, y_train
    
    # Auto strategy: balance to equal class sizes
    smote = SMOTE(
        sampling_strategy='auto',  # Balance to 1:1 ratio
        k_neighbors=k_neighbors,
        random_state=SEED
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
    print(f"After SMOTE: {resampled_counts.to_dict()}")
    print(f"Generated {len(y_resampled) - len(y_train)} synthetic samples")
    
    return X_resampled, y_resampled


# Hyperparameter Optimization


def optimize_hyperparameters(X_train, y_train, config):
    """
    Use Hyperopt with proper cross-validation for small datasets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        best_params: Optimal hyperparameters
        best_score: Best CV score achieved
    """
    print("\n Hyperparameter Optimization ")
    print(f"Using {config['cv_folds']}-fold stratified CV")
    print(f"Max evaluations: {config['hyperopt_evals']}")
    
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
    
    best_score_global = {'score': -np.inf}
    
    def objective(params):
        # Convert to integers where needed
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, 
                             random_state=SEED)
        
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
        
        # Update best score
        if mean_f1 > best_score_global['score']:
            best_score_global['score'] = mean_f1
        
        print(f"Trial | Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
        
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
    best_trial = min(trials.results, key=lambda x: x['loss'])
    best_params = best_trial['params']
    
    print(f"\n Optimization Complete!")
    print(f"Best CV F1: {best_trial['mean_f1']:.4f} ± {best_trial['std_f1']:.4f}")
    print(f"Best params: {best_params}")
    
    return best_params, best_trial['mean_f1']


# Model Training and Evaluation
def train_and_evaluate(best_params, X_train, X_test, y_train, y_test):
    """
    Train final model and perform comprehensive evaluation.
    
    Args:
        best_params: Optimal hyperparameters from Hyperopt
        X_train, X_test: Feature matrices
        y_train, y_test: Labels
        
    Returns:
        clf: Trained classifier
        metrics: Dictionary of performance metrics
    """
    print("\n Training Final Model ")
    
    # Train final model
    clf = XGBClassifier(
        **best_params,
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False
    )
    clf.fit(X_train, y_train, verbose=False)
    print(" Model trained successfully")
    
    # Predictions
    print("\n Model Evaluation ")
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
    
    # Print results
    print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy:  {test_acc*100:.2f}%")
    print(f"Test F1 Score:  {test_f1:.4f}")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    
    # Check for overfitting
    if train_acc - test_acc > 0.15:
        print("\n Warning: Possible overfitting detected (train-test gap > 15%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, 
                               target_names=["Control", "ASD"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_prob_test, test_auc)
    
    return clf, metrics

def plot_roc_curve(y_test, y_prob, auc_score):
    """Generate and display ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, 
            label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", 
            label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - XGBoost Classifier", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# SHAP Analysis


def run_shap_analysis(model, X_train, X_test, y_test, feature_names):
    """
    Generate SHAP explanations for model interpretability.
    
    Args:
        model: Trained XGBoost classifier
        X_train: Training features (for background)
        X_test: Test features
        y_test: Test labels
        feature_names: Gene names
    """
    print("\n SHAP Analysis ")
    print("  Generating SHAP values (this may take a minute)...")
    
    # Convert to DataFrames with gene names
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Use subset for faster computation
    background = X_train_df.sample(n=min(50, len(X_train_df)), random_state=SEED)
    
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test_df)
    
    # Summary plot (bar)
    print("   Generating feature importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()
    
    # Summary plot (beeswarm)
    print("   Generating SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.tight_layout()
    plt.show()
    
    # Waterfall plots for example predictions
    if len(y_test) > 0:
        # Control example
        control_indices = np.where(y_test.values == 0)[0]
        if len(control_indices) > 0:
            print("   Generating waterfall plot (Control example)...")
            _create_waterfall(shap_values, explainer, X_test_df, 
                            control_indices[0], "Control Sample")
        
        # ASD example
        asd_indices = np.where(y_test.values == 1)[0]
        if len(asd_indices) > 0:
            print("   Generating waterfall plot (ASD example)...")
            _create_waterfall(shap_values, explainer, X_test_df, 
                            asd_indices[0], "ASD Sample")

def _create_waterfall(shap_values, explainer, X_test, idx, title):
    """Helper to create waterfall plot."""
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_test.iloc[idx],
            feature_names=X_test.columns
        ),
        max_display=15,
        show=False
    )
    plt.title(f"SHAP Waterfall Plot - {title}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


# Single Sample Prediction Demo
def predict_single_sample(clf, X_test, y_test, feature_names, sample_idx=0):
    """Demonstrate prediction on a single test sample."""
    print("\n Single Sample Prediction Demo ")
    
    if sample_idx >= len(X_test):
        sample_idx = 0
        print(f"Adjusted sample_idx to {sample_idx}")
    
    X_single = X_test[sample_idx].reshape(1, -1)
    y_true = y_test.iloc[sample_idx]
    
    y_pred = clf.predict(X_single)[0]
    y_prob = clf.predict_proba(X_single)[0]
    
    print(f"\nSample Index: {sample_idx}")
    print(f"True Label:      {y_true} ({'ASD' if y_true == 1 else 'Control'})")
    print(f"Predicted Label: {y_pred} ({'ASD' if y_pred == 1 else 'Control'})")
    print(f"Confidence:")
    print(f"- Control: {y_prob[0]:.1%}")
    print(f"- ASD:     {y_prob[1]:.1%}")
    
    if y_pred == y_true:
        print("Correct prediction")
    else:
        print("Incorrect prediction")


# Main Pipeline
def main():
    """Execute complete ML pipeline with proper RNA-seq handling."""
    
    print("\n" + "="*70)
    print("  ML Classification Pipeline - ASD vs Control (RNA-Seq Data)")
    print("="*70)
    
    # Setup
    set_seed(SEED)
    
    # Step 1: Load and preprocess
    X, y, feature_names = load_and_preprocess("data/ML_dataset.csv")
    
    # Step 2: Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], stratify=y, random_state=SEED
    )
    print(f"\n Split: {len(X_train)} train | {len(X_test)} test samples")
    
    # Step 3: Feature selection BEFORE SMOTE (CRITICAL FIX)
    X_train_sel, X_test_sel, selected_features = select_features(
        X_train.values, y_train.values, X_test.values, 
        feature_names, CONFIG
    )
    
    # Step 4: Handle class imbalance with SMOTE (AFTER feature selection)
    X_train_balanced, y_train_balanced = handle_imbalance(
        X_train_sel, y_train.values, CONFIG
    )
    
    # Step 5: Hyperparameter optimization
    best_params, best_cv_score = optimize_hyperparameters(
        X_train_balanced, y_train_balanced, CONFIG
    )
    
    # Step 6: Train final model and evaluate
    clf, metrics = train_and_evaluate(
        best_params, X_train_balanced, X_test_sel, 
        y_train_balanced, y_test
    )
    
    # Step 7: SHAP interpretability analysis
    run_shap_analysis(clf, X_train_balanced, X_test_sel, y_test, 
                     selected_features)
    
    # Step 8: Demo single prediction
    predict_single_sample(clf, X_test_sel, y_test, selected_features, 
                         sample_idx=0)
    
    print("\n" + "="*70)
    print("  Pipeline Complete! ")
    print("="*70)
    print(f"\nFinal Performance Summary:")
    print(f"CV F1 Score:    {best_cv_score:.4f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']*100:.2f}%")
    print(f"Test F1 Score:  {metrics['test_f1']:.4f}")
    print(f"Test ROC-AUC:   {metrics['test_auc']:.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
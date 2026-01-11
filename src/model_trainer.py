"""
Model Training and Evaluation Module

Provides utilities for training, evaluating, and comparing
multiple machine learning models for churn prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class ChurnModelTrainer:
    """Training and evaluation utilities for churn prediction models."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        
    def get_models(self) -> Dict[str, Any]:
        """Return dictionary of models to train."""
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=2,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                class_weight='balanced',
                verbose=-1
            )
        }
        return models
    
    def apply_sampling(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply sampling to handle class imbalance."""
        
        if method == 'smote':
            sampler = SMOTE(random_state=42)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'combined':
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            sampler = ImbPipeline(steps=[('o', over), ('u', under)])
        else:
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"Sampling applied ({method}): {len(y)} -> {len(y_resampled)} samples")
        print(f"  New churn rate: {y_resampled.mean():.2%}")
        
        return X_resampled, y_resampled
    
    def train_model(
        self, 
        model: Any, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Any:
        """Train a single model."""
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance."""
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'avg_precision': average_precision_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def cross_validate(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5
    ) -> Dict:
        """Perform cross-validation."""
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
            'f1': cross_val_score(model, X, y, cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        }
        
        return {
            'accuracy_mean': scores['accuracy'].mean(),
            'accuracy_std': scores['accuracy'].std(),
            'f1_mean': scores['f1'].mean(),
            'f1_std': scores['f1'].std(),
            'roc_auc_mean': scores['roc_auc'].mean(),
            'roc_auc_std': scores['roc_auc'].std()
        }
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        use_sampling: str = None
    ) -> Dict[str, Dict]:
        """Train and evaluate all models."""
        
        if use_sampling:
            X_train_sampled, y_train_sampled = self.apply_sampling(
                X_train, y_train, method=use_sampling
            )
        else:
            X_train_sampled, y_train_sampled = X_train, y_train
        
        models = self.get_models()
        results = {}
        
        print("\nModel Training")
        print("=" * 70)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            trained_model = self.train_model(model, X_train_sampled, y_train_sampled)
            self.models[name] = trained_model
            
            metrics = self.evaluate_model(trained_model, X_test, y_test)
            cv_scores = self.cross_validate(model, X_train_sampled, y_train_sampled)
            metrics['cv_scores'] = cv_scores
            
            results[name] = metrics
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")
            print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"  CV ROC AUC: {cv_scores['roc_auc_mean']:.4f} +/- {cv_scores['roc_auc_std']:.4f}")
        
        self.results = results
        print("\n" + "=" * 70)
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Generate model comparison table."""
        
        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'ROC AUC': metrics['roc_auc'],
                'CV ROC AUC': metrics['cv_scores']['roc_auc_mean']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('ROC AUC', ascending=False)
        
        return df
    
    def select_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any]:
        """Select the best performing model."""
        
        best_score = 0
        best_name = ""
        
        for name, metrics in self.results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest model: {best_name}")
        print(f"  {metric.upper()}: {best_score:.4f}")
        
        return best_name, self.best_model
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from the best model."""
        
        if self.best_model is None:
            self.select_best_model()
        
        model = self.best_model
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str, model: Any = None):
        """Save model to file."""
        if model is None:
            model = self.best_model
        
        joblib.dump(model, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, filepath: str) -> Any:
        """Load model from file."""
        model = joblib.load(filepath)
        print(f"Model loaded: {filepath}")
        return model
    
    def predict_churn(
        self, 
        model: Any, 
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate churn predictions."""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        return predictions, probabilities


def print_classification_report(y_true: pd.Series, y_pred: np.ndarray):
    """Print detailed classification report."""
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(
        y_true, y_pred, 
        target_names=['No Churn', 'Churn']
    ))


if __name__ == "__main__":
    print("Model trainer module loaded.")

"""
Model Explainability Module

Provides SHAP-based explanations for model predictions
and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Any, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ChurnExplainer:
    """SHAP-based model explainability tools."""
    
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background: pd.DataFrame, explainer_type: str = 'tree'):
        """Create SHAP explainer."""
        
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(X_background, 100)
            )
        elif explainer_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model, X_background)
        else:
            self.explainer = shap.Explainer(self.model, X_background)
            
        print(f"SHAP {explainer_type} explainer created")
        
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for the dataset."""
        
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification output
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]
        
        # Handle 3D arrays
        if isinstance(self.shap_values, np.ndarray) and len(self.shap_values.shape) == 3:
            self.shap_values = self.shap_values[:, :, 1]
        
        print(f"SHAP values calculated: {self.shap_values.shape}")
        return self.shap_values
    
    def plot_summary(self, X: pd.DataFrame, max_display: int = 15, save_path: str = None):
        """Generate SHAP summary plot."""
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved: {save_path}")
        
        plt.show()
        
    def plot_bar(self, X: pd.DataFrame, max_display: int = 15, save_path: str = None):
        """Generate SHAP bar plot."""
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.title('Mean |SHAP Value|', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def explain_instance(
        self, 
        X: pd.DataFrame, 
        instance_idx: int,
        save_path: str = None
    ):
        """Generate waterfall plot for a single instance."""
        
        shap_explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X.iloc[instance_idx].values,
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, show=False)
        plt.title(f'SHAP Explanation - Instance {instance_idx}', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_force(self, X: pd.DataFrame, instance_idx: int):
        """Generate force plot for a single instance."""
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[1]
        
        shap.force_plot(
            base_value,
            self.shap_values[instance_idx],
            X.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=True
        )
    
    def plot_dependence(
        self, 
        X: pd.DataFrame, 
        feature: str, 
        interaction_feature: str = None,
        save_path: str = None
    ):
        """Generate feature dependence plot."""
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, 
            self.shap_values, 
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence: {feature}', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_top_features(self, top_n: int = 10) -> pd.DataFrame:
        """Get top important features based on SHAP values."""
        
        shap_vals = self.shap_values
        
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        
        mean_shap = np.abs(shap_vals).mean(axis=0)
        
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=False).head(top_n)
        
        return importance_df
    
    def explain_high_risk_customers(
        self, 
        X: pd.DataFrame, 
        probabilities: np.ndarray,
        top_n: int = 5
    ) -> Dict:
        """Analyze highest risk customers."""
        
        high_risk_idx = np.argsort(probabilities)[-top_n:][::-1]
        
        explanations = {}
        for idx in high_risk_idx:
            instance_shap = self.shap_values[idx]
            top_features_idx = np.argsort(np.abs(instance_shap))[-5:][::-1]
            
            explanations[idx] = {
                'churn_probability': probabilities[idx],
                'top_factors': [
                    {
                        'feature': self.feature_names[i],
                        'value': X.iloc[idx, i],
                        'shap_value': instance_shap[i],
                        'direction': 'increases' if instance_shap[i] > 0 else 'decreases'
                    }
                    for i in top_features_idx
                ]
            }
        
        return explanations


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix."""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, save_path: str = None):
    """Plot Precision-Recall curve."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    print("Explainability module loaded.")

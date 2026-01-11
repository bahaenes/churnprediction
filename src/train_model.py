"""
Model Training Script

Command-line script for training the churn prediction model.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import ChurnDataProcessor
from src.model_trainer import ChurnModelTrainer
import joblib


def main():
    print("=" * 60)
    print("CHURN MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    print("\n[1/5] Loading data...")
    processor = ChurnDataProcessor()
    df = processor.load_data(str(project_root / "data" / "telco_churn.csv"))
    
    print("\n[2/5] Preparing data...")
    X_train, X_test, y_train, y_test = processor.prepare_data(df, test_size=0.2, random_state=42)
    
    # Train models
    print("\n[3/5] Training models...")
    trainer = ChurnModelTrainer()
    results = trainer.train_all_models(
        X_train, y_train,
        X_test, y_test,
        use_sampling='smote'
    )
    
    # Compare models
    print("\n[4/5] Model Comparison:")
    comparison = trainer.compare_models()
    print(comparison.to_string(index=False))
    
    # Select and save best model
    best_name, best_model = trainer.select_best_model(metric='roc_auc')
    best_results = results[best_name]
    
    # Save artifacts
    print("\n[5/5] Saving model artifacts...")
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    trainer.save_model(str(models_dir / "best_model.joblib"))
    processor.save_processor(str(models_dir / "processor.joblib"))
    
    model_results = {
        'model_name': best_name,
        'accuracy': best_results['accuracy'],
        'precision': best_results['precision'],
        'recall': best_results['recall'],
        'f1': best_results['f1'],
        'roc_auc': best_results['roc_auc'],
        'feature_columns': X_train.columns.tolist()
    }
    joblib.dump(model_results, str(models_dir / "model_results.joblib"))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {best_name}")
    print(f"ROC AUC: {best_results['roc_auc']:.4f}")
    print(f"F1 Score: {best_results['f1']:.4f}")
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"\nArtifacts saved to: {models_dir}")


if __name__ == "__main__":
    main()

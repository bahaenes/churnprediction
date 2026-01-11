# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn in the telecommunications industry.

## Project Overview

This project implements a complete data science pipeline to predict which customers are likely to churn (cancel their service). It includes data preprocessing, feature engineering, model training, evaluation, and explainability analysis.

## Project Structure

```
churn_prediction/
├── app/
│   └── streamlit_app.py      # Interactive web dashboard
├── data/
│   └── telco_churn.csv       # Dataset (generated on first run)
├── models/
│   ├── best_model.joblib     # Trained model
│   ├── processor.joblib      # Data preprocessor
│   └── model_results.joblib  # Model metrics
├── notebooks/
│   └── churn_analysis.ipynb  # Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Synthetic data generator
│   ├── preprocessing.py      # Data preprocessing pipeline
│   ├── model_trainer.py      # Model training utilities
│   ├── explainability.py     # SHAP analysis tools
│   └── train_model.py        # Training script
├── requirements.txt
├── .gitignore
└── README.md
```

## Features

### Data Processing
- Automated data cleaning and validation
- Feature engineering with 10+ derived features
- Categorical encoding and numerical scaling
- Train-test split with stratification

### Feature Engineering
- `TenureGroup` - Customer tenure categories
- `ChargeGroup` - Monthly charge segments
- `TotalServices` - Count of subscribed services
- `HasSecurityServices` - Security service indicator
- `HasStreamingServices` - Streaming service indicator
- `AutoPayment` - Automatic payment flag
- `AvgMonthlySpend` - Average spending per month
- `RiskScore` - Calculated churn risk score

### Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### Evaluation
- Cross-validation with stratified K-fold
- ROC AUC, F1 Score, Precision, Recall metrics
- Confusion matrix visualization
- ROC and Precision-Recall curves

### Explainability
- SHAP (SHapley Additive exPlanations) analysis
- Global feature importance
- Individual prediction explanations
- High-risk customer profiling

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the analysis notebook

```bash
cd notebooks
jupyter notebook churn_analysis.ipynb
```

### Option 2: Train model via command line

```bash
python src/train_model.py
```

### Option 3: Launch the web dashboard

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` in your browser.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.725 | 0.669 | 0.727 | 0.697 | 0.801 |
| Random Forest | 0.727 | 0.656 | 0.783 | 0.714 | 0.802 |
| Gradient Boosting | 0.723 | 0.660 | 0.747 | 0.701 | 0.794 |
| XGBoost | 0.677 | 0.584 | 0.893 | 0.706 | 0.795 |
| LightGBM | 0.718 | 0.654 | 0.745 | 0.697 | 0.791 |

**Best Model:** Random Forest with ROC AUC of 0.802

### Top Features (SHAP Analysis)

| Feature | Importance |
|---------|------------|
| RiskScore | 0.070 |
| Contract_Two year | 0.052 |
| tenure | 0.040 |
| Contract_One year | 0.030 |
| TotalCharges | 0.028 |

## Technologies

- **Python 3.10+**
- **Pandas, NumPy** - Data manipulation
- **Scikit-learn** - Machine learning
- **XGBoost, LightGBM** - Gradient boosting
- **imbalanced-learn** - SMOTE for class imbalance
- **SHAP** - Model interpretability
- **Streamlit** - Web application
- **Plotly, Seaborn, Matplotlib** - Visualization

## License

MIT License

## Author

[Your Name]

---

*This project was developed as part of a data science portfolio.*

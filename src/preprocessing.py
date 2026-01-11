"""
Data Preprocessing and Feature Engineering Module

Provides utilities for cleaning, transforming, and preparing
customer data for churn prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import joblib
from pathlib import Path


class ChurnDataProcessor:
    """Data preprocessing and feature engineering for churn prediction."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def explore_data(self, df: pd.DataFrame) -> Dict:
        """Generate basic statistics for data exploration."""
        stats = {
            'shape': df.shape,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'churn_distribution': df['Churn'].value_counts().to_dict(),
            'churn_rate': (df['Churn'] == 'Yes').mean()
        }
        return stats
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        df = df.copy()
        
        # Convert TotalCharges to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing values
        mask = df['TotalCharges'].isnull()
        df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
        
        # Remove customerID (not needed for modeling)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        print(f"Data cleaned. Missing values: {df.isnull().sum().sum()}")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        df = df.copy()
        
        # Tenure groups
        df['TenureGroup'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72],
            labels=['0-1 Year', '1-2 Years', '2-4 Years', '4+ Years']
        )
        
        # Charge groups
        df['ChargeGroup'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 35, 55, 75, 120],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Total services count
        def count_services(row):
            count = 0
            if row.get('PhoneService') == 'Yes': count += 1
            if row.get('MultipleLines') == 'Yes': count += 1
            if row.get('InternetService') != 'No': count += 1
            for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                       'TechSupport', 'StreamingTV', 'StreamingMovies']:
                if row.get(col) == 'Yes': count += 1
            return count
        
        df['TotalServices'] = df.apply(count_services, axis=1)
        
        # Security services flag
        df['HasSecurityServices'] = (
            (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
        ).astype(int)
        
        # Streaming services flag
        df['HasStreamingServices'] = (
            (df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')
        ).astype(int)
        
        # Auto payment flag
        df['AutoPayment'] = df['PaymentMethod'].isin([
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ]).astype(int)
        
        # Average monthly spend
        df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # Services per tenure
        df['ServicesPerTenure'] = df['TotalServices'] / (df['tenure'] + 1)
        
        # High value customer flag
        df['HighValueCustomer'] = (
            (df['tenure'] > 24) & (df['MonthlyCharges'] > 70)
        ).astype(int)
        
        # Risk score
        df['RiskScore'] = 0
        df.loc[df['Contract'] == 'Month-to-month', 'RiskScore'] += 2
        df.loc[df['tenure'] < 12, 'RiskScore'] += 2
        df.loc[df['PaymentMethod'] == 'Electronic check', 'RiskScore'] += 1
        df.loc[df['HasSecurityServices'] == 0, 'RiskScore'] += 1
        df.loc[df['MonthlyCharges'] > 80, 'RiskScore'] += 1
        
        print(f"Feature engineering complete. Total columns: {len(df.columns)}")
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        self.categorical_columns = categorical_cols
        
        binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        
        for col in categorical_cols:
            unique_vals = df[col].unique()
            
            if set(unique_vals).issubset({'Yes', 'No'}):
                df[col] = df[col].map(binary_map)
            elif set(unique_vals).issubset({'Male', 'Female'}):
                df[col] = df[col].map(binary_map)
            else:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        print(f"Encoding complete. Total columns: {len(df.columns)}")
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                     'TotalServices', 'AvgMonthlySpend', 'ServicesPerTenure']
        
        scale_cols = [col for col in scale_cols if col in df.columns]
        self.numerical_columns = scale_cols
        
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])
        
        print(f"Scaling complete: {scale_cols}")
        return df
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute the complete data preparation pipeline."""
        
        df = self.clean_data(df)
        df = self.create_features(df)
        df = self.encode_features(df, fit=True)
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                     'TotalServices', 'AvgMonthlySpend', 'ServicesPerTenure']
        scale_cols = [col for col in scale_cols if col in X_train.columns]
        
        X_train[scale_cols] = self.scaler.fit_transform(X_train[scale_cols])
        X_test[scale_cols] = self.scaler.transform(X_test[scale_cols])
        
        self.feature_columns = X_train.columns.tolist()
        
        print(f"\nData Preparation Summary:")
        print(f"  Train set: {X_train.shape[0]} samples")
        print(f"  Test set: {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Train churn rate: {y_train.mean():.2%}")
        print(f"  Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processor(self, filepath: str):
        """Save the processor state."""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns
        }, filepath)
        print(f"Processor saved: {filepath}")
    
    def load_processor(self, filepath: str):
        """Load the processor state."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.categorical_columns = data['categorical_columns']
        self.numerical_columns = data['numerical_columns']
        print(f"Processor loaded: {filepath}")


if __name__ == "__main__":
    processor = ChurnDataProcessor()
    df = processor.load_data("data/telco_churn.csv")
    stats = processor.explore_data(df)
    print(f"Churn rate: {stats['churn_rate']:.2%}")

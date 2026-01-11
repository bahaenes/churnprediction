"""
Telco Customer Churn Dataset Generator

Generates a realistic synthetic dataset for customer churn prediction
in the telecommunications industry.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_churn_dataset(n_samples: int = 7000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic telco customer churn dataset.
    
    Parameters
    ----------
    n_samples : int, default=7000
        Number of samples to generate.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Generated customer dataset with churn labels.
    """
    np.random.seed(random_state)
    
    # Customer IDs
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, n_samples + 1)]
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Tenure (months) - exponential distribution
    tenure = np.clip(np.random.exponential(scale=32, size=n_samples), 1, 72).astype(int)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    
    multiple_lines = np.where(
        phone_service == 'Yes',
        np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58]),
        'No phone service'
    )
    
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], 
        n_samples, 
        p=[0.34, 0.44, 0.22]
    )
    
    def internet_dependent_service(has_internet, prob_yes=0.5):
        return np.where(
            has_internet != 'No',
            np.random.choice(['Yes', 'No'], n_samples, p=[prob_yes, 1-prob_yes]),
            'No internet service'
        )
    
    online_security = internet_dependent_service(internet_service, 0.29)
    online_backup = internet_dependent_service(internet_service, 0.34)
    device_protection = internet_dependent_service(internet_service, 0.34)
    tech_support = internet_dependent_service(internet_service, 0.29)
    streaming_tv = internet_dependent_service(internet_service, 0.38)
    streaming_movies = internet_dependent_service(internet_service, 0.39)
    
    # Contract
    contract = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.55, 0.21, 0.24]
    )
    
    # Billing
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Monthly charges calculation
    base_charge = 20
    monthly_charges = base_charge + np.zeros(n_samples)
    
    monthly_charges += np.where(phone_service == 'Yes', 20, 0)
    monthly_charges += np.where(multiple_lines == 'Yes', 10, 0)
    monthly_charges += np.where(internet_service == 'DSL', 25, 0)
    monthly_charges += np.where(internet_service == 'Fiber optic', 45, 0)
    monthly_charges += np.where(online_security == 'Yes', 5, 0)
    monthly_charges += np.where(online_backup == 'Yes', 5, 0)
    monthly_charges += np.where(device_protection == 'Yes', 5, 0)
    monthly_charges += np.where(tech_support == 'Yes', 5, 0)
    monthly_charges += np.where(streaming_tv == 'Yes', 10, 0)
    monthly_charges += np.where(streaming_movies == 'Yes', 10, 0)
    
    monthly_charges += np.random.uniform(-5, 5, n_samples)
    monthly_charges = np.round(monthly_charges, 2)
    monthly_charges = np.clip(monthly_charges, 18, 120)
    
    # Total charges
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(-50, 50, n_samples), 2)
    total_charges = np.clip(total_charges, 0, None)
    
    # Churn probability calculation (realistic patterns)
    churn_prob = np.zeros(n_samples)
    
    # Base churn rate
    churn_prob += 0.15
    
    # Tenure effect
    churn_prob += np.where(tenure < 12, 0.25, 0)
    churn_prob += np.where(tenure < 6, 0.15, 0)
    churn_prob -= np.where(tenure > 48, 0.15, 0)
    
    # Contract effect
    churn_prob += np.where(contract == 'Month-to-month', 0.25, 0)
    churn_prob -= np.where(contract == 'Two year', 0.20, 0)
    churn_prob -= np.where(contract == 'One year', 0.10, 0)
    
    # Internet service effect
    churn_prob += np.where(internet_service == 'Fiber optic', 0.10, 0)
    
    # Security services effect
    churn_prob -= np.where(online_security == 'Yes', 0.08, 0)
    churn_prob -= np.where(tech_support == 'Yes', 0.08, 0)
    churn_prob -= np.where(online_backup == 'Yes', 0.05, 0)
    
    # Payment method effect
    churn_prob += np.where(payment_method == 'Electronic check', 0.10, 0)
    
    # Demographics
    churn_prob += np.where(senior_citizen == 1, 0.05, 0)
    churn_prob += np.where(paperless_billing == 'Yes', 0.05, 0)
    
    # High charges effect
    churn_prob += np.where(monthly_charges > 80, 0.10, 0)
    
    # Normalize probabilities
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_prob).astype(str)
    churn = np.where(churn == '1', 'Yes', 'No')
    
    # Create DataFrame
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df


def main():
    """Generate and save the dataset."""
    print("Generating churn dataset...")
    
    df = generate_churn_dataset(n_samples=7000, random_state=42)
    
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = data_dir / "telco_churn.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Dataset saved: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    
    return df


if __name__ == "__main__":
    main()

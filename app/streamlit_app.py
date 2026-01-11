"""
Customer Churn Prediction Dashboard

Interactive Streamlit application for predicting customer churn
and exploring model insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_PATH = PROJECT_ROOT / "models"


@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    model = joblib.load(MODELS_PATH / "best_model.joblib")
    processor = joblib.load(MODELS_PATH / "processor.joblib")
    results = joblib.load(MODELS_PATH / "model_results.joblib")
    return model, processor, results


def predict_churn(model, processor, customer_data):
    """Generate churn prediction for a single customer."""
    df = pd.DataFrame([customer_data])
    X_processed = processor.transform(df)
    
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]
    
    return prediction, probability


def main():
    st.title("Customer Churn Prediction Dashboard")
    
    # Load artifacts
    try:
        model, processor, results = load_model_artifacts()
        model_loaded = True
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        model_loaded = False
        return
    
    # Sidebar - Model Performance
    st.sidebar.header("Model Information")
    st.sidebar.write(f"**Model:** {results['model_name']}")
    st.sidebar.write(f"**ROC AUC:** {results['roc_auc']:.4f}")
    st.sidebar.write(f"**F1 Score:** {results['f1']:.4f}")
    st.sidebar.write(f"**Accuracy:** {results['accuracy']:.4f}")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Individual Customer Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col3:
            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
            total_charges = monthly_charges * tenure
            st.write(f"**Total Charges:** ${total_charges:.2f}")
        
        # Prepare customer data
        customer_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
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
            'TotalCharges': total_charges
        }
        
        # Prediction button
        if st.button("Predict Churn", type="primary"):
            prediction, probability = predict_churn(model, processor, customer_data)
            
            st.divider()
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.error(f"### High Churn Risk")
                    st.write("This customer is likely to churn.")
                else:
                    st.success(f"### Low Churn Risk")
                    st.write("This customer is likely to stay.")
            
            with col_result2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Churn Probability (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors
            st.subheader("Key Risk Factors")
            
            risk_factors = []
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract (higher churn risk)")
            if tenure < 12:
                risk_factors.append(f"Short tenure ({tenure} months)")
            if internet_service == "Fiber optic" and online_security == "No":
                risk_factors.append("Fiber optic without online security")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment method")
            if monthly_charges > 70:
                risk_factors.append(f"High monthly charges (${monthly_charges:.2f})")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("No significant risk factors identified.")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        st.write("Upload a CSV file with customer data to get bulk predictions.")
        
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records")
                st.dataframe(df.head())
                
                if st.button("Generate Predictions", type="primary"):
                    with st.spinner("Processing..."):
                        X_processed = processor.transform(df)
                        predictions = model.predict(X_processed)
                        probabilities = model.predict_proba(X_processed)[:, 1]
                        
                        df['Churn_Prediction'] = predictions
                        df['Churn_Probability'] = probabilities
                        df['Risk_Level'] = pd.cut(
                            probabilities,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Medium', 'High']
                        )
                    
                    st.success("Predictions complete!")
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    
                    churn_rate = predictions.mean() * 100
                    high_risk = (df['Risk_Level'] == 'High').sum()
                    avg_prob = probabilities.mean() * 100
                    
                    col1.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
                    col2.metric("High Risk Customers", high_risk)
                    col3.metric("Average Churn Probability", f"{avg_prob:.1f}%")
                    
                    # Distribution chart
                    fig = px.histogram(
                        df, x='Churn_Probability',
                        nbins=20,
                        title='Distribution of Churn Probabilities'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("Prediction Results")
                    st.dataframe(df)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")


if __name__ == "__main__":
    main()

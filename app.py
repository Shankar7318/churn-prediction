import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load all saved models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'knn': 'knn_model.pkl',
        'svm': 'svm_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'adaboost': 'adaboost_model.pkl',
        'gradient_boosting': 'gradient_boosting_model.pkl',
        'voting_classifier': 'voting_classifier_model.pkl'
    }
    
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'rb') as file:
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            st.error(f"Model file {file_path} not found. Please make sure all model files are in the same directory.")
    
    return models

def preprocess_input(customer_data):
    """Preprocess the input data to match training format"""
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([customer_data])
    
    # Map categorical variables to numerical (based on typical encoding)
    gender_map = {'Female': 0, 'Male': 1}
    yes_no_map = {'No': 0, 'Yes': 1}
    internet_service_map = {'DSL': 0, 'Fiber optic': 1, 'No': 2}
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_method_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
    
    # Apply mappings
    input_df['gender'] = input_df['gender'].map(gender_map)
    
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_columns:
        input_df[col] = input_df[col].map(yes_no_map)
    
    multiple_lines_map = {'No phone service': 0, 'No': 1, 'Yes': 2}
    input_df['MultipleLines'] = input_df['MultipleLines'].map(multiple_lines_map)
    
    input_df['InternetService'] = input_df['InternetService'].map(internet_service_map)
    
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_map = {'No internet service': 0, 'No': 1, 'Yes': 2}
    for col in service_columns:
        input_df[col] = input_df[col].map(service_map)
    
    input_df['Contract'] = input_df['Contract'].map(contract_map)
    input_df['PaymentMethod'] = input_df['PaymentMethod'].map(payment_method_map)
    
    # Convert TotalCharges to numeric
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    
    return input_df

def main():
    st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
    
    # Title and description
    st.title("📊 Customer Churn Prediction App")
    st.markdown("""
    This app predicts customer churn probability using multiple machine learning models.
    Fill in the customer details below and select a model to get predictions.
    """)
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No models loaded. Please check if all model files are available.")
        return
    
    # Create two columns for input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", 
            "Mailed check", 
            "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        # Service options (only show if internet service is selected)
        if internet_service != "No":
            online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        else:
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"
        
        st.subheader("Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    
    # Model selection
    st.subheader("Model Selection")
    selected_model = st.selectbox(
        "Choose a prediction model:",
        list(models.keys())
    )
    
    # Prediction button
    if st.button("Predict Churn", type="primary"):
        # Prepare input data
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
        
        try:
            # Preprocess input
            input_df = preprocess_input(customer_data)
            
            # Get selected model
            model = models[selected_model]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    label="Churn Prediction", 
                    value="Yes" if prediction == 1 else "No",
                    delta="High Risk" if prediction == 1 else "Low Risk"
                )
            
            with result_col2:
                churn_probability = prediction_proba[1] * 100
                st.metric(
                    label="Churn Probability", 
                    value=f"{churn_probability:.1f}%"
                )
            
            with result_col3:
                no_churn_probability = prediction_proba[0] * 100
                st.metric(
                    label="No Churn Probability", 
                    value=f"{no_churn_probability:.1f}%"
                )
            
            # Probability visualization
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['No Churn', 'Churn'],
                'Probability': [no_churn_probability, churn_probability]
            })
            
            st.bar_chart(prob_df.set_index('Outcome'))
            
            # Interpretation
            st.subheader("Interpretation")
            if prediction == 1:
                st.error("🚨 This customer has a high probability of churning. Consider retention strategies.")
                if churn_probability > 70:
                    st.warning("⚠️ Very high churn risk! Immediate action recommended.")
                elif churn_probability > 50:
                    st.warning("⚠️ High churn risk! Proactive measures needed.")
            else:
                st.success("✅ This customer has a low probability of churning.")
                if churn_probability < 20:
                    st.info("💡 Very low churn risk. Customer appears stable.")
                else:
                    st.info("💡 Moderate churn risk. Regular monitoring recommended.")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                feature_importance = pd.DataFrame({
                    'feature': input_df.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                st.bar_chart(feature_importance.set_index('feature')['importance'].head(10))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check if the input data is in the correct format.")
    
    # Model comparison section
    st.subheader("Model Comparison")
    if st.button("Compare All Models"):
        try:
            # Prepare input data (using default values or current form values)
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
            
            input_df = preprocess_input(customer_data)
            
            comparison_data = []
            for model_name, model in models.items():
                try:
                    prediction = model.predict(input_df)[0]
                    prediction_proba = model.predict_proba(input_df)[0]
                    churn_prob = prediction_proba[1] * 100
                    
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Prediction': 'Yes' if prediction == 1 else 'No',
                        'Churn Probability (%)': round(churn_prob, 2),
                        'Confidence': 'High' if abs(churn_prob - 50) > 30 else 'Medium'
                    })
                except Exception as e:
                    st.warning(f"Could not get prediction from {model_name}: {str(e)}")
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Visualize comparison
                st.bar_chart(comparison_df.set_index('Model')['Churn Probability (%)'])
        
        except Exception as e:
            st.error(f"Error in model comparison: {str(e)}")
    
    # Sidebar with additional information
    st.sidebar.title("About")
    st.sidebar.info("""
    This churn prediction app uses multiple machine learning models trained on customer data to predict the likelihood of customer churn.
    
    **Available Models:**
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Logistic Regression
    - Decision Tree
    - AdaBoost
    - Gradient Boosting
    - Voting Classifier (Ensemble)
    
    Fill in the customer information and select a model to get predictions.
    """)
    
    st.sidebar.title("Model Information")
    st.sidebar.write(f"**Loaded Models:** {len(models)}")
    st.sidebar.write("**Available Models:**")
    for model_name in models.keys():
        st.sidebar.write(f"- {model_name.replace('_', ' ').title()}")

if __name__ == "__main__":
    main()
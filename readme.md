# 📊 Customer Churn Prediction App

A comprehensive **Streamlit web application** for predicting customer churn using multiple machine learning models.  
This app helps businesses identify customers at risk of leaving and take proactive retention measures.

---

## 🚀 Features

- **Multiple Model Support:** Choose from 7 different ML models  
- **Real-time Predictions:** Get instant churn predictions with probability scores  
- **Interactive Interface:** User-friendly form for customer data input  
- **Visual Analytics:** Charts and graphs for better insights  
- **Model Comparison:** Compare predictions across all available models  
- **Feature Importance:** Understand which factors drive churn predictions  
- **Risk Assessment:** Clear interpretation of prediction results  

---

## 📋 Available Models

- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Logistic Regression  
- Decision Tree  
- AdaBoost  
- Gradient Boosting  
- Voting Classifier (Ensemble)  

---

## 🛠️ Installation

### **Prerequisites**
- Python 3.7+
- pip (Python package manager)

### **Step 1: Clone the Repository**

git clone <your-repository-url>
cd churn-prediction-app

### **step 2: Step 2: Create Virtual Environment**
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

### **Step 3: Install Dependencies
pip install -r requirements.txt

# Required Model Files
```
Ensure the following trained model files are in the project root directory:

knn_model.pkl
svm_model.pkl
logistic_regression_model.pkl
decision_tree_model.pkl
adaboost_model.pkl
gradient_boosting_model.pkl
voting_classifier_model.pkl
```

# 🎯 Usage
Run the Application
streamlit run app.py


The app will open in your browser at http://localhost:8501

# How to Use

- Input Customer Data: Fill in demographic, account, service, and charge information.

- Select Model: Choose from the available ML models.

- Get Prediction: Click “Predict Churn” to view:

- Churn prediction (Yes/No)

- Probability scores

- Risk assessment

- Visual probability distribution

- Compare Models: Use the “Compare All Models” feature to see predictions side-by-side.

# 📊 Input Features
```Demographics

Gender

Senior Citizen status

Partner status

Dependents

Account Information

Tenure (months)

Contract type

Paperless billing

Payment method

Services

Phone service

Multiple lines

Internet service type

Online security

Online backup

Device protection

Tech support

Streaming TV

Streaming movies

Charges

Monthly charges

Total charges
```

# 🔧 Project Structure
```
churn-prediction-app/
│
├── app.py                       # Main Streamlit application
├── requirements.txt              # Python dependencies
├── knn_model.pkl                 # K-Nearest Neighbors model
├── svm_model.pkl                 # Support Vector Machine model
├── logistic_regression_model.pkl
├── decision_tree_model.pkl
├── adaboost_model.pkl
├── gradient_boosting_model.pkl
├── voting_classifier_model.pkl
└── README.md                     # Project documentation
```

# 📈 Model Training

The machine learning models were trained on customer churn data with the following process:

Data Preprocessing: Handling missing values, encoding categorical variables

Feature Engineering: Creating relevant features for prediction

Model Training: Training multiple algorithms with hyperparameter tuning

Model Evaluation: Using metrics like accuracy, precision, recall, and F1-score

Model Serialization: Saving trained models as pickle files for deployment

# 🎨 Application Sections
Main Input Form

Two-column layout for organized data entry

Dynamic form fields based on selections

Real-time validation and error handling

Prediction Results

Clear churn prediction (Yes/No)

Probability scores with visual charts

Risk level assessment

Actionable insights and recommendations

Model Comparison

Side-by-side comparison of all models

Consistency analysis across algorithms

Confidence level indicators

Feature Importance

Visual representation of important features

Insights into what drives churn predictions

# 🚨 Interpretation Guide
```Churn Probability Ranges
Range	Risk Level	Action
0–20%	Very Low	Customer appears stable
20–40%	Low	Regular monitoring recommended
40–60%	Moderate	Proactive engagement suggested
60–80%	High	Retention measures needed
80–100%	Very High	Immediate action required
Risk Levels


Low Risk: Customer retention likely

Medium Risk: Monitor and engage periodically

High Risk: Implement retention strategies

Critical Risk: Immediate intervention required
```

# 🔍 Technical Details
- Preprocessing Pipeline

- Categorical variable encoding

- Numerical feature scaling

- Missing value handling

- Feature transformation matching training process

# Model Loading

- Cached model loading for performance

- Error handling for missing model files

- Supports various scikit-learn model types

- Prediction Engine

- Real-time data preprocessing

- Probability calibration

- Confidence scoring

- Ensemble model support

# 🐛 Troubleshooting
Common Issues

1. Model Files Not Found

Error: Model file not found


💡 Solution: Ensure all .pkl files are in the project root directory.

2. Dependency Errors

Error: Module not found


💡 Solution: Run:

pip install -r requirements.txt


3. Port Already in Use

Error: Port 8501 already in use


💡 Solution: Use:

streamlit run app.py --server.port 8502


4. Prediction Errors

Error: Shape mismatch or preprocessing error


💡 Solution: Check if input data matches the training format.

Performance Tips

Use Chrome or Firefox for best results

Close unused tabs to improve responsiveness

Ensure sufficient RAM for model loading

Use virtual environment for dependency isolation

📝 License

This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributing

Contributions are welcome!
```To contribute:

Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request
```
# 📞 Support

For help or questions:

- Open an issue on GitHub

Check the troubleshooting section

Review the model training notebook for dataset details

# 🔮 Future Enhancements

- Model performance metrics dashboard

- Batch prediction capabilities

- Customer segmentation analysis

- Retention strategy recommendations

- Historical prediction tracking

- API endpoint for integration

- Advanced visualization options

- Multi-language support
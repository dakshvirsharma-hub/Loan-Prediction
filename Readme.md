🏦 Loan Eligibility Prediction
A machine learning classification project that predicts whether a loan application will be approved (Y) or rejected (N).
📌 Project Overview
Built an end-to-end ML workflow from data preprocessing to Streamlit deployment.
🔄 Workflow
Data → Cleaning → EDA → Missing Values → Encoding → Train/Test Split → Model Training → Evaluation → GridSearchCV → Random Forest → Model Saving → Streamlit
📊 Features
- Gender
- Married
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Amount Term
- Credit History
- Property Area
loan_id was removed because it is an identifier.
🤖 Model
Random Forest Classifier
- n_estimators = 100
- max_depth = 5
- min_samples_split = 5
- min_samples_leaf = 2
- Hyperparameter tuning: GridSearchCV
📈 Performance
Metric	Score
Accuracy	79.8%
F1 Score	86.3%
Recall	98.5%
ROC-AUC	78.4%


🔧 Preprocessing
- Missing-value handling
- Label Encoding for binary categorical features
- One-Hot Encoding for property_area
- Feature/target separation
- Consistent feature ordering during prediction
🌐 Streamlit App
The application accepts customer information and provides:
- Loan Status: Y / N
- Approval Probability
💾 Saved Files
loan_prediction_model.joblib
label_encoder.joblib
onehot_encoder.joblib
🛠️ Tech Stack
Python • Pandas • NumPy • Scikit-learn • Matplotlib • Seaborn • Joblib • Streamlit
▶️ Run
pip install -r requirements.txt
streamlit run app.py

## 📸 Project Screenshot

![Loan Prediction App](Loanprediction.png)# Loan-Prediction
# Loan-Prediction

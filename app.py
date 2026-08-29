import streamlit as st
import pandas as pd
import joblib
import base64


with open("background.png", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Load model and encoders
model = joblib.load('loan_prediction_model.joblib')
ohe = joblib.load('onehot_encoder.joblib')

model = joblib.load('loan_prediction_model.joblib')
le = joblib.load('label_encoders.joblib')
ohe= joblib.load('onehot_encoder.joblib')


# Two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.title('Loan Eligibility Prediction using Machine Learning')
    st.write('Enter customer details to predict loan eligibility')

with col2:
    st.title('Model Information')
    st.write('Model :- Random Forest')
    st.write('n_estimators = 100')
    st.write('max_depth = 5')
    st.write('min_sample_split = 5')
    st.write('min_sample_leaf = 2')
    st.write('Accuracy = 79.8%')
    st.write('F1 Score = 86.3%')
    st.write('Recall = 98.5%')
    st.write('ROC-AUC = 78.4%')


# Customer Information
st.header("Customer Profile")

gender = st.selectbox(
    "Gender",
    ["male", "female"]
)

married = st.selectbox(
    "Married",
    ["yes", "no"]
)

dependents = st.slider(
    "Dependents",
    min_value=0,
    max_value=3,
    value=0,
    step=1
)

education = st.selectbox(
    "Education",
    ["graduate", "not graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["yes", "no"]
)


# Financial Information
st.header("Financial Information")

Applicantincome = st.number_input(
    "Applicant Income",
    min_value=0
)

Coapplicantincome = st.number_input(
    "Coapplicant Income",
    min_value=0
)

Loanamount = st.number_input(
    "Loan Amount",
    min_value=0
)

Loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0
)


# Credit Information
st.header("Credit Information")

credit_history = st.slider(
    "Credit History",
    min_value=0,
    max_value=1,
    value=1,
    step=1
)


# Property Information
st.header("Property Information")

property_information = st.selectbox(
    "Property Area",
    ["urban", "rural", "semiurban"]
)


# Prediction
if st.button("Predict eligibility"):
    input_data=pd.DataFrame({
        "gender" : [gender],
        "married" : [married],
        "dependents" : [dependents],
        "education" : [education],
        "self_employed" :[self_employed],
        "applicantincome" : [Applicantincome],
        "coapplicantincome" : [Coapplicantincome],
        "loanamount" :[Loanamount],
        "loan_amount_term" : [Loan_amount_term],
        "credit_history" : [credit_history],
        "property_area" : [property_information]
    })

    label_encoder = [
        "gender",
        "married",
        "education",
        "self_employed"
    ]
    for col in label_encoder:
        input_data[col]= le[col].transform(
            input_data[col]
        )

    multi_col =['property_area']

    encoded = ohe.transform(input_data[multi_col])
    encoded_df = pd.DataFrame(
        encoded,
        columns = ohe.get_feature_names_out(multi_col)
    )

    input_data = input_data.drop(
        columns=multi_col
    )

    final_data = pd.concat(
        [input_data,encoded_df],
        axis=1
    )

    columns = [
    'gender',
    'married',
    'dependents',
    'education',
    'self_employed',
    'applicantincome',
    'coapplicantincome',
    'loanamount',
    'loan_amount_term',
    'credit_history',
    'property_area_rural',
    'property_area_semiurban',
    'property_area_urban'
    ]

    final_data = final_data[columns]

    predict = model.predict_proba(final_data)

    probability = predict[0][1]

    if probability >= 0.5:
        result = (st.success("🟢Yes Loan Eligbility Success"))
    else:
        result = (st.error("🔴 No Loan Eligbility does not success"))


    st.write(f"Approval Probability: {probability:.2%}")
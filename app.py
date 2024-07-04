import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

loan_model = pickle.load(
    open(
        "loan_model.sav",
        "rb",
    )
)

medical_insurance = pickle.load(
    open(
        "medical_insurance.sav",
        "rb",
    )
)

spam_model = pickle.load(
    open(
        "spam_model.sav",
        "rb",
    )
)

titanic_model = pickle.load(
    open(
        "titanic_model.sav",
        "rb",
    )
)

vectorizer_model = pickle.load(open("vec.sav", "rb"))

st.set_page_config(page_title="Predictive Insights", page_icon="🔮", layout="wide")

with st.sidebar:
    selected = option_menu(
        "ML Predictive Insights",
        [
            "Index",
            "Loan Status Prediction",
            "Medical Insurance Cost Prediction",
            "Spam Mail Prediction",
            "Titanic Survival Prediction",
            "Connect"
        ],
        icons=["bookmark", "piggy-bank", "coin", "envelope", "life-preserver", "person-lines-fill"],
        default_index=0,
    )

if selected == "Index":
    st.title("Machine Learning Predictive Insights")
    st.markdown("---")
    st.subheader(" Explore our models designed to predict various outcomes:")
    st.write(
        "- Loan Status Prediction: Predict whether a loan will be approved or rejected."
    )
    st.write(
        "- Medical Insurance Cost Prediction: Estimate medical insurance costs based on individual factors."
    )
    st.write(
        "- Spam Mail Prediction: Determine if an email is likely to be spam or normal."
    )
    st.write(
        "- Titanic Survival Prediction: Predict survival chances on the Titanic based on passenger details."
    )
    st.write("Select a prediction task from the sidebar to get started.")
    st.markdown("""
    <div style='color: gray; font-size: 15px;'>
        Last updated on 29/06/2024.
    </div>
    """, unsafe_allow_html=True)

if selected == "Loan Status Prediction":
    st.title("Loan Status Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        g = st.text_input("Gender: Female-0 Male-1")
        e = st.text_input("Education: Not Graduate-0 Graduate-1")
        c = st.text_input("Co-applicant Income")
        ch = st.text_input("Credit History")

    with col2:
        m = st.text_input("Married: No-0 Yes-1")
        s = st.text_input("Self Employed: No-0 Yes-1")
        l = st.text_input("Loan Amount")
        p = st.text_input("Property Area")

    with col3:
        d = st.text_input("Number of Dependents")
        a = st.text_input("Applicant Income")
        lt = st.text_input("Loan Amount Term")

    loan_status = ""

    if st.button("Loan Status Result"):

        user_input = [g, m, d, e, s, a, c, l, lt, ch, p]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)

        loan_prediction = loan_model.predict(user_input)

        if loan_prediction[0] == 1:
            loan_status = "Loan Approved"
        else:
            loan_status = "Loan Rejected"

    st.success(loan_status)

if selected == "Medical Insurance Cost Prediction":
    st.title("Medical Insurance Cost Prediction")

    a = st.text_input("Age")
    s = st.text_input("Sex")
    b = st.text_input("BMI")
    ch = st.text_input("Children")
    sm = st.text_input("Smoker")
    r = st.text_input("Region")

    insurance_cost = ""

    if st.button("Medical Insurance Cost"):

        user_input = [a, s, b, ch, sm, r]
        user_input = np.array([float(x) for x in user_input]).reshape(1, -1)

        insurance_cost = medical_insurance.predict(user_input)

    st.success(f"The insurance cost is: ${insurance_cost}")


if selected == "Spam Mail Prediction":
    st.title("Spam Mail Prediction")

    spam_status = " "
    inp = st.text_input("Enter Email")
    if st.button("Spam Mail Predictor"):

        inp_transformed = vectorizer_model.transform([inp])

        prediction = spam_model.predict(inp_transformed)
        if prediction[0] == 1:
            spam_status = "Normal Email"
        else:
            spam_status = "Spam Email"

    st.success(spam_status)

if selected == "Titanic Survival Prediction":
    st.title("Titanic Survival Prediction")

    p = st.text_input("Ticket Class")
    s = st.selectbox("Sex", ["Male", "Female"])
    a = st.text_input("Age")
    sibsp = st.text_input("Number of siblings/spouses aboard")
    parch = st.text_input("Number of parents/children aboard")
    f = st.text_input("Passenger Fare")
    e = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    titanic_survival = ""

    if st.button("Titanic Survival Predictor"):
        p = float(p)
        a = float(a)
        sibsp = float(sibsp)
        parch = float(parch)
        f = float(f)

        sex_encoded = 1 if s == "Female" else 0
        embark_encoded = {"C": 1, "Q": 2, "S": 0}[e]

        user_input = np.array(
            [p, sex_encoded, a, sibsp, parch, f, embark_encoded]
        ).reshape(1, -1)

        survival = titanic_model.predict(user_input)

        if survival[0] == 1:
            titanic_survival = "Alive"
        else:
            titanic_survival = "Dead"

    st.success(f"Predicted Survival: {titanic_survival}")

if selected == "Connect":
    import streamlit as st

    st.title("Connect with Me")
    st.markdown("""
        Feel free to reach out to me via email or connect with me on LinkedIn and GitHub! 💬
    """)
    st.write("")

    st.subheader("Contact Information 📩")
    st.write("**Name:** Aryan Shah")
    st.write("**Email:** aryanshah1957@gmail.com")
    st.write("")

    st.subheader("Social Media 🌍")
    st.write("[LinkedIn](https://www.linkedin.com/in/aryanashah/) 🔗")
    st.write("[GitHub](https://github.com/AryanShah30) 🔗")

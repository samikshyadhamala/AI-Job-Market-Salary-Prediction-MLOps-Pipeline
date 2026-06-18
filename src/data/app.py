import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="",
    layout="centered",
)

st.title("AI Job Salary Prediction")

with st.form("salary_prediction_form"):
    st.subheader("Role Details")

    job_title = st.text_input(
        "Job Title",
        "Machine Learning Engineer",
    )

    industry = st.text_input(
        "Industry",
        "Technology",
    )

    required_skills_text = st.text_area(
        "Required Skills",
        "Python, Machine Learning, SQL",
    )

    education_required = st.selectbox(
        "Education Required",
        ["Associate", "Bachelor", "Master", "PhD"],
        index=2,
    )

    st.subheader("Experience And Work Type")

    years_experience = st.number_input(
        "Years Experience",
        min_value=0,
        max_value=50,
        value=5,
    )

    experience_level = st.selectbox(
        "Experience Level ('EN': 'entry_level','MI': 'mid_level','SE': 'senior','EX': 'executive')",
        ["EN", "MI", "SE", "EX"],
        index=1,
    )

    employment_type = st.selectbox(
        "Employment Type (full time, part time, contract ,freelance)",
        ["FT", "PT", "CT", "FL"],
    )

    remote_ratio = st.selectbox(
        "Remote Ratio",
        [0, 50, 100],
        index=2,
    )

    st.subheader("Company And Location")

    company_location = st.text_input(
        "Company Location",
        "United States",
    )

    employee_residence = st.text_input(
        "Employee Residence",
        "United States",
    )

    company_size = st.selectbox(
        "Company Size",
        ["S", "M", "L"],
        index=1,
    )

    st.text_input("Salary Currency", value="USD", disabled=True)
    salary_currency = "USD"

    benefits_score = st.slider(
        "Benefits Score",
        min_value=0.0,
        max_value=10.0,
        value=8.5,
    )

    days_open = st.number_input(
        "Days Open",
        min_value=0,
        max_value=3650,
        value=30,
    )

    submitted = st.form_submit_button("Predict Salary")


if submitted:
    required_skills = [
        skill.strip()
        for skill in required_skills_text.split(",")
        if skill.strip()
    ]

    payload = {
        "years_experience": int(years_experience),
        "remote_ratio": int(remote_ratio),
        "benefits_score": float(benefits_score),
        "job_description_length": 1500,
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "company_location": company_location,
        "employee_residence": employee_residence,
        "company_size": company_size,
        "education_required": education_required,
        "industry": industry,
        "salary_currency": salary_currency,
        "required_skills": required_skills,
        "days_open": int(days_open),
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            st.error(f"Prediction failed: {detail}")
        else:
            result = response.json()
            st.success(
                f"Predicted Salary: ${result['predicted_salary_usd']:,.2f}"
            )
            st.caption(
                f"Model version: {result.get('model_version', 'unknown')} | "
                f"From cache: {result.get('from_cache', False)}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. Start the deployment server on "
            "http://127.0.0.1:8000 first."
        )
    except requests.exceptions.Timeout:
        st.error("Prediction request timed out. Please try again.")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

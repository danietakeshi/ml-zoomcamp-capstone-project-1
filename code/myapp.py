import streamlit as st
import requests
import sys

# Function to make the API call
def make_api_call(data):
    # Replace this URL with your API endpoint
    url = f"{sys.argv[1]}/predict" if len(sys.argv) > 1 else "http://0.0.0.0:9696/predict"
    
    # Making the API call
    response = requests.post(url, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        st.success("API call successful!")

        answer = response.json()

        st.write("Customer ", f'has a chance of churn with a probability of {answer["churn_probability"]:.2f}%' if answer['churn_flag'] else f'will not churn with a probability of {answer["churn_probability"]:.2f}%')
    else:
        st.error("Failed to make the API call.")

def main():
    st.title("Customer Credit Card Churn Check")
    
    # Input fields for customer data
    customer_age = st.number_input("Customer Age", min_value=0, max_value=150, value=48)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    dependent_count = st.number_input("Dependent Count", min_value=0, value=2)
    education_level = st.selectbox("Education Level", ["High School", "Graduate", "Post-Graduate", "Doctorate", "Uneducated", "College"], index=1)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
    income_category = st.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"], index=2)
    card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"], index=1)
    months_on_book = st.number_input("Months on Book", min_value=0, value=35)
    total_relationship_count = st.number_input("Total Relationship Count", min_value=0, value=2)
    months_inactive_12_mon = st.number_input("Months Inactive (12 months)", min_value=0, value=4)
    contacts_count_12_mon = st.number_input("Contacts Count (12 months)", min_value=0, value=4)
    credit_limit = st.number_input("Credit Limit", min_value=0, value=34516)
    total_revolving_bal = st.number_input("Total Revolving Balance", min_value=0, value=0)
    total_amt_chng_q4_q1 = st.number_input("Total Amount Change Q4-Q1", min_value=0.0, value=0.763, format="%.2f")
    total_trans_amt = st.number_input("Total Transaction Amount", min_value=0.0, value=691.0, format="%.2f")
    
    # Button to trigger the API call
    if st.button("Make API Call"):
        # Store input data in a dictionary
        data = {
            "customer_age": customer_age,
            "gender": gender,
            "dependent_count": dependent_count,
            "education_level": education_level,
            "marital_status": marital_status,
            "income_category": income_category,
            "card_category": card_category,
            "months_on_book": months_on_book,
            "total_relationship_count": total_relationship_count,
            "months_inactive_12_mon": months_inactive_12_mon,
            "contacts_count_12_mon": contacts_count_12_mon,
            "credit_limit": credit_limit,
            "total_revolving_bal": total_revolving_bal,
            "total_amt_chng_q4_q1": total_amt_chng_q4_q1,
            "total_trans_amt": total_trans_amt
        }
        make_api_call(data)

if __name__ == "__main__":
    main()

import requests
import sys

if len(sys.argv) == 1:
    url = 'http://0.0.0.0:9696/predict'
else:
    url = f'{sys.argv[1]}/predict'

test_churn = [
    {
    "customer_age":62,
    "gender":"F",
    "dependent_count":0,
    "education_level":"Graduate",
    "marital_status":"Married",
    "income_category":"Less than $40K",
    "card_category":"Blue",
    "months_on_book":49,
    "total_relationship_count":2,
    "months_inactive_12_mon":3,
    "contacts_count_12_mon":3,
    "credit_limit":1438.3,
    "total_revolving_bal":0,
    "total_amt_chng_q4_q1":1.047,
    "total_trans_amt":692
    },

    {
    "customer_age":54,
    "gender":"F",
    "dependent_count":1,
    "education_level":"Graduate",
    "marital_status":"Married",
    "income_category":"Less than $40K",
    "card_category":"Blue",
    "months_on_book":40,
    "total_relationship_count":2,
    "months_inactive_12_mon":3,
    "contacts_count_12_mon":1,
    "credit_limit":1438.3,
    "total_revolving_bal":808,
    "total_amt_chng_q4_q1":0.997,
    "total_trans_amt":705
    },

    {
    "customer_age":56,
    "gender":"M",
    "dependent_count":2,
    "education_level":"Graduate",
    "marital_status":"Married",
    "income_category":"$120K +",
    "card_category":"Blue",
    "months_on_book":36,
    "total_relationship_count":1,
    "months_inactive_12_mon":3,
    "contacts_count_12_mon":3,
    "credit_limit":15769.0,
    "total_revolving_bal":0,
    "total_amt_chng_q4_q1":1.041,
    "total_trans_amt":602
    },

    {
    "customer_age":48,
    "gender":"M",
    "dependent_count":2,
    "education_level":"Graduate",
    "marital_status":"Married",
    "income_category":"$60K - $80K",
    "card_category":"Silver",
    "months_on_book":35,
    "total_relationship_count":2,
    "months_inactive_12_mon":4,
    "contacts_count_12_mon":4,
    "credit_limit":34516.0,
    "total_revolving_bal":0,
    "total_amt_chng_q4_q1":0.763,
    "total_trans_amt":691
    }
]

test_no_churn = [
    {
    "customer_age":45,
    "gender":"M",
    "dependent_count":3,
    "education_level":"High School",
    "marital_status":"Married",
    "income_category":"$60K - $80K",
    "card_category":"Blue",
    "months_on_book":39,
    "total_relationship_count":5,
    "months_inactive_12_mon":1,
    "contacts_count_12_mon":3,
    "credit_limit":12691.0,
    "total_revolving_bal":777,
    "total_amt_chng_q4_q1":1.335,
    "total_trans_amt":1144
    },

    {
    "customer_age":49,
    "gender":"F",
    "dependent_count":5,
    "education_level":"Graduate",
    "marital_status":"Single",
    "income_category":"Less than $40K",
    "card_category":"Blue",
    "months_on_book":44,
    "total_relationship_count":6,
    "months_inactive_12_mon":1,
    "contacts_count_12_mon":2,
    "credit_limit":8256.0,
    "total_revolving_bal":864,
    "total_amt_chng_q4_q1":1.541,
    "total_trans_amt":1291
    },

    {
    "customer_age":51,
    "gender":"M",
    "dependent_count":3,
    "education_level":"Graduate",
    "marital_status":"Married",
    "income_category":"$80K - $120K",
    "card_category":"Blue",
    "months_on_book":36,
    "total_relationship_count":4,
    "months_inactive_12_mon":1,
    "contacts_count_12_mon":0,
    "credit_limit":3418.0,
    "total_revolving_bal":0,
    "total_amt_chng_q4_q1":2.594,
    "total_trans_amt":1887
    },

    {
    "customer_age":40,
    "gender":"F",
    "dependent_count":4,
    "education_level":"High School",
    "marital_status":"Unknown",
    "income_category":"Less than $40K",
    "card_category":"Blue",
    "months_on_book":34,
    "total_relationship_count":3,
    "months_inactive_12_mon":4,
    "contacts_count_12_mon":1,
    "credit_limit":3313.0,
    "total_revolving_bal":2517,
    "total_amt_chng_q4_q1":1.405,
    "total_trans_amt":1171
    }
]
print("=========================== CHURNING CUSTOMERS TEST ===========================")
for enum, test_case in enumerate(test_churn):
    answer = requests.post(url, json=test_case).json()
    print(f"Response {enum}: {answer}")

print("========================= NON CHURNING CUSTOMERS TEST ===========================")
for enum, test_case in enumerate(test_no_churn):
    answer = requests.post(url, json=test_case).json()
    print(f"Response {enum}: {answer}")

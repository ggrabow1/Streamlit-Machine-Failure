# -*- coding: utf-8 -*-
"""
Created on 6 - 07 - 2022

@authors: Gene Grabowski, Jr., Sean T Ford

Uses streamlit to create an application. Calls Intelligent Decisioning in Viya
to run a decion flow including an XGB model and recommender system.
"""

#import os
#import sys
import pandas as pd
import numpy as np
import streamlit as st
#import keyring
import requests
import json

#custom post that provides Viya authentication (OAuth2) with http request
#Note - requires an admin to create a token for user
def post(url1, contentType, accept, accessToken, body):
    sess = requests.Session()
    headers = {"Accept": accept,
    "Authorization": "bearer " + accessToken,
    "Content-Type": contentType }
    # Convert the request body to a JSON object.
    reqBody = json.loads(body)
    # Post the request.
    req = sess.post(url1, json=reqBody, headers=headers)
    sess.close()
    return req;

#generate inputs in the format ID is expecting from a dictionary
def gen_viya_inputs(feature_dict):
    feature_list = []
    for k,v in feature_dict.items():
        if type(v) == str:
            feature_list.append(f'{{"name": "{k}_", "value" : "{v}"}}')
        else:
            feature_list.append(f'{{"name": "{k}_", "value" : {v}}}')
            
    feature_str = str.join(',',feature_list)
    
    return '{"inputs" : [' + feature_str + ']}'

#call the ID API and get the results as a python dictionary
def call_id_api(baseUrl, accessToken, feature_dict,moduleID):
    #create the response = call_id_api(baseUrl, token, features, moduleID)
        print(response['outputs'])
        output_dict = unpack_viya_outputs(response['outputs']) in format viya wants
    requestBody = gen_viya_inputs(feature_dict)

    # Define the content and accept types for the request header.
    contentType = "application/json"
    acceptType = "application/json"
    
    # Define the request URL.
    masModuleUrl = "/microanalyticScore/modules/" + moduleID
    requestUrl = baseUrl + masModuleUrl + "/steps/execute"
    
    # Execute the decision.
    print(requestBody)
    masExecutionResponse = post(requestUrl, contentType,
     acceptType, accessToken, requestBody)
    # Display the response.
    return json.loads(masExecutionResponse.content)

#unpack the ID outputs section as a python dictionary
def unpack_viya_outputs(outputs):
    d = {}
    for elem in outputs:
        d[elem['name']] = '' if 'value' not in elem.keys() else elem['value']
        
    return d
        

#authentication token (get from Paige)
token='eyJhbGciOiJSUzI1NiIsImprdSI6Imh0dHBzOi8vbG9jYWxob3N0L1NBU0xvZ29uL3Rva2VuX2tleXMiLCJraWQiOiJsZWdhY3ktdG9rZW4ta2V5IiwidHlwIjoiSldUIn0.eyJqdGkiOiJmZTQxMWRkNTI1YWU0ZWU0OWM3YWQ5YmFmYjcxYjVjMSIsImV4dF9pZCI6InVpZD1nZWdyYWIsb3U9dXNlcnMsZGM9YnVkcyxkYz1jb20iLCJyZW1vdGVfaXAiOiIxNzIuMTYuNDguMTc2Iiwic3ViIjoiNDdjMzFjZTctOGZhMC00YmE1LWI5YmMtNjk0NDQ1NzFiNTQwIiwic2NvcGUiOlsib3BlbmlkIl0sImNsaWVudF9pZCI6Im15Y2xpZW50aWQiLCJjaWQiOiJteWNsaWVudGlkIiwiYXpwIjoibXljbGllbnRpZCIsImdyYW50X3R5cGUiOiJhdXRob3JpemF0aW9uX2NvZGUiLCJ1c2VyX2lkIjoiNDdjMzFjZTctOGZhMC00YmE1LWI5YmMtNjk0NDQ1NzFiNTQwIiwib3JpZ2luIjoibGRhcCIsInVzZXJfbmFtZSI6ImdlZ3JhYiIsImVtYWlsIjoiZ2VncmFiQGVlY2x4dm0wNjguZXhuZXQuc2FzLmNvbSIsImF1dGhfdGltZSI6MTY1NDgwMTMyOCwicmV2X3NpZyI6ImIwNWJiM2E2IiwiaWF0IjoxNjU0ODAxMzI4LCJleHAiOjE5NzAxNjEzMjgsImlzcyI6Imh0dHA6Ly9sb2NhbGhvc3QvU0FTTG9nb24vb2F1dGgvdG9rZW4iLCJ6aWQiOiJ1YWEiLCJhdWQiOlsibXljbGllbnRpZCIsIm9wZW5pZCJdfQ.OjRxqIr92x-A7dakoQMbcduROHT5ts6T6i1FWOrzPAa_qO2aYPERKbdKzLhgdmkI0vCHdS8vqIKVgiml0qe0HGjAyP0j2iZ0hsx8nlpaNJ4hJls7-vldHKINLf_3EdstEOwgcGWZqGQh9UV3sntUb5-bU0F3jbZYh6zOTNWHhmy6SNHzJUJzB77RYsuzYTopuNnQDt_WU9Y3lj4AOKeqNbOv6IGDWQEpunQYqoISoe4dCasKbrTyiFQqYnLmDqwr_KLt4C7A8pRRCpsqRg7OQVnViCrOME_1C6t9j917g5DXc366cDfH7nK24Enb6Heh42qW03P9tXq8VlJh8SI5Kg'

host='eeclxvm067.exnet.sas.com'
protocol='http'

#base URL for Viya
baseUrl = protocol + '://' + host + '/'

#insert your model ID for the model saved in MAS
moduleID = "machine failure gg1_combo3"


st.title('Machine Failure Prediction App')
st.write("This web app classifies machines with a probability of failure.")
st.write("Models:  **Extreme Gradient Boosting and Neural Networks**")
st.write("Python Packages:  **XGBoost with ONNX and PyTorch with ONNX**")



df = pd.read_csv('Data/machine_failure.csv')

df.drop(['FAILURE'], axis=1, inplace=True)
df.REASON.replace(np.nan, 'TruckPr', regex=True, inplace=True)
df.INDUSTRY.replace(np.nan, 'Other', regex=True, inplace=True)
df=df.fillna(0)



machine_id_filter = st.selectbox("Select the Machine ID", pd.unique(df['MACHINE_ID']))
df = df[df['MACHINE_ID']==machine_id_filter]

# print(df.Loan_ID)
MACHINE_ID = int(df.MACHINE_ID)




DAILY_PRODUCTION = st.sidebar.slider(label = 'Avg Amount of Parts Produced Daily', min_value = 0,
                          max_value = 30000 ,
                          value =int(df.DAILY_PRODUCTION),
                          step = 100)

MONTHLY_PRODUCTION = st.sidebar.slider(label = 'Monthly Part Production', min_value = 2000,
                          max_value = 40000 ,
                          value = int(df.MONTHLY_PRODUCTION),
                          step = 1000)

PROD_TARGET = st.sidebar.slider(label = 'Monthly Production Target', min_value = 10000,
                          max_value = 90000 ,
                          value = int(df.PROD_TARGET),
                          step = 1000)

#REASON = st.sidebar.selectbox('Reason for Loan',['CarImp','DebtCon'])
REASON = st.sidebar.selectbox('Type of Production',df.REASON)



MONTHS_IN_OPERATION = st.sidebar.slider(label = 'Months in Operation for Machine', min_value = 0,
                          max_value = 40,
                          value = int(df.MONTHS_IN_OPERATION),
                          step = 1)


NEGATIVE_RATING = st.sidebar.slider(label = 'Negative Production Reports', min_value = 0,
                          max_value = 10,
                          value = int(df.NEGATIVE_RATING),
                          step = 1)

DAYS_OUT_OF_SERVICE = st.sidebar.slider(label = 'Days Out of Service', min_value = 0,
                          max_value = 200,
                          value = int(df.DAYS_OUT_OF_SERVICE),
                          step = 1)

MACHINE_AGE = st.sidebar.slider(label = 'Age of Machine', min_value = 0,
                          max_value = 15,
                          value = int(df.MACHINE_AGE),
                          step = 1)
SENSOR_INQUIRIES = st.sidebar.slider(label = 'Number of Sensor Inquiries', min_value = 0,
                          max_value = 20,
                          value = int(df.SENSOR_INQUIRIES),
                          step = 1)

NUM_SENSORS = st.sidebar.slider(label = 'Number of Sensors', min_value = 0,
                          max_value = 50,
                          value = int(df.NUM_SENSORS),
                          step = 1)

SENSOR_RATIO = st.sidebar.slider(label = 'Sensor Ratio', min_value = 0,
                          max_value = 100,
                          value = int(df.SENSOR_RATIO),
                          step = 1)


#JOB = st.sidebar.selectbox('Job Type',['Other','ProfExe','Office', 'Mgr', 'Self', 'Sales'])
INDUSTRY = st.sidebar.selectbox('Type of Industry',df.INDUSTRY)

######





features = {'MACHINE_ID': MACHINE_ID,
            'DAILY_PRODUCTION': DAILY_PRODUCTION,
            'MONTHLY_PRODUCTION': MONTHLY_PRODUCTION,
            'PROD_TARGET': PROD_TARGET,
            'REASON':  REASON,
            'MONTHS_IN_OPERATION':  MONTHS_IN_OPERATION,
            'NEGATIVE_RATING':  NEGATIVE_RATING,
            'DAYS_OUT_OF_SERVICE':  DAYS_OUT_OF_SERVICE,
            'MACHINE_AGE': MACHINE_AGE,
            'SENSOR_INQUIRIES':  SENSOR_INQUIRIES,
            'NUM_SENSORS':  NUM_SENSORS,
            'SENSOR_RATIO':  SENSOR_RATIO,
            'INDUSTRY':  INDUSTRY
           }



features_df  = pd.DataFrame([features])


def format(x):
    return "{:,.0f}".format(x)

#def format2(x):
    #return "{:.0%}".format(x/100)



features_df['DAILY_PRODUCTION'] = features_df['DAILY_PRODUCTION'].apply(format)
features_df['MONTHLY_PRODUCTION'] = features_df['MONTHLY_PRODUCTION'].apply(format)
features_df['PROD_TARGET'] = features_df['PROD_TARGET'].apply(format)
# features_df['LOANDUE'] = features_df['LOANDUE'].apply(format)
# features_df['VALUE'] = features_df['VALUE'].apply(format)
# features_df['DEBTINC'] = features_df['DEBTINC'].apply(format2)


features_df = features_df.rename(columns={'DAILY_PRODUCTION': 'Avg Amount of Parts Produced Daily',
                                          'MONTHLY_PRODUCTION': 'Monthly Part Production',
                                          'PROD_TARGET': 'Monthly Production Target',
                                          'REASON': 'Type of Production',
                                          'MONTHS_IN_OPERATION': 'Months in Operation for Machine',
                                          'NEGATIVE_RATING': 'Negative Production Reports',
                                          'DAYS_OUT_OF_SERVICE': 'Days Out of Service',
                                          'MACHINE_AGE': 'Age of Machine',
                                          'SENSOR_INQUIRIES': 'Number of Sensor Inquiries',
                                          'NUM_SENSORS': 'Number of Sensors',
                                          'SENSOR_RATIO': 'Sensor Ratio',
                                          'INDUSTRY': 'Type of Industry',
                                          })



hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.table(features_df)





if st.button('Predict'):
    
    response = call_id_api(baseUrl, token, features, moduleID)
    output_dict = unpack_viya_outputs(response['outputs'])







    if (output_dict['P_FAILURE'] >= 0.5):


        status = '*** High Likelihood to Fail ***'
    else:
        status = 'Low Likelihood to Fail'

    if (output_dict['ACTION_1'] == ''):
        action = 'None'
    else:
        action = output_dict['ACTION_1']
       
    st.write("Machine Status:")
    st.write(status)
    st.write ("Probability of Failure is:")
    st.write (f"{output_dict['P_FAILURE']:.2%}")
    st.write("Recommended action: ")
    st.write(action)

    
 
    

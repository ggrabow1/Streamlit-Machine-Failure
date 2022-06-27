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
    #create the request in format viya wants
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
#moduleID = "machine anomaly gg recSys max_jun2022"
moduleID = "Machine Anomaly GG RecSys MAX_JUN2022"

st.title('Machine Anomaly Detection App')
st.write("This web app predicts anomalies and provides a recommendation for a next best action.")
st.write("Indicates which sensor contributes most to the anomalous state.")
st.write("Models:  **Deep Learning Autoencoders**")
st.write("Python Packages:  **Tensorflow and Keras**")



df = pd.read_csv('Data/machine_anomaly_test_1.csv')



#df.drop(['TARGET'], axis=1, inplace=True)





machine_id_filter = st.selectbox("Select the Machine ID", pd.unique(df['SENSOR_DATE']))
df = df[df['SENSOR_DATE']==machine_id_filter]


# print(df.Loan_ID)
#MACHINE_ID = df.MACHINE_ID




S_0 = st.sidebar.slider(label = 'S_0', min_value =-10.0,
                          max_value = 5.0,
                          value =float(df.S_0),
                          step = 0.1)

 

######





#features = {'S_0': S_0
#           }



features_df  = pd.DataFrame(df)
print(features_df)


#def format(x):
#    return "{:,.0f}".format(x)

#def format2(x):
    #return "{:.0%}".format(x/100)



# features_df['DAILY_PRODUCTION'] = features_df['DAILY_PRODUCTION'].apply(format)
# features_df['MONTHLY_PRODUCTION'] = features_df['MONTHLY_PRODUCTION'].apply(format)
# features_df['PROD_TARGET'] = features_df['PROD_TARGET'].apply(format)
# features_df['LOANDUE'] = features_df['LOANDUE'].apply(format)
# features_df['VALUE'] = features_df['VALUE'].apply(format)
# features_df['DEBTINC'] = features_df['DEBTINC'].apply(format2)


#features_df = features_df.rename(columns={'S_0': 'S_0'
#                                          })



hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
#st.markdown(hide_table_row_index, unsafe_allow_html=True)

#st.table(features_df)





if st.button('Predict'):
    
    response = call_id_api(baseUrl, token, features_df, moduleID)
    #print(response)
    #output_dict = unpack_viya_outputs(response['outputs'])







    # if (output_dict['P_FAILURE'] >= 0.5):
    #
    #
    #     status = '*** High Likelihood to Fail ***'
    # else:
    #     status = 'Low Likelihood to Fail'
    #
    # if (output_dict['ACTION_1'] == ''):
    #     action = 'None'
    # else:
    #     action = output_dict['ACTION_1']
       
    st.write("Machine Status:")
    #st.write(status)
    st.write ("Probability of Failure is:")
    #st.write ({output_dict['ANOMALY_FLAG']})
    st.write("Recommended action: ")
      
    #st.write(action)

    
 
    

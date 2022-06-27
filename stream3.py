
import streamlit as st  # web development
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import time  # to simulate a real time data, time loop
import plotly.express as px  # interactive charts
import requests
import json



def post(url1, contentType, accept, accessToken, body):
    sess = requests.Session()
    headers = {"Accept": accept,
    "Authorization": "bearer " + accessToken,
    "Content-Type": contentType}
    # Convert the request body to a JSON object.
    reqBody = json.loads(body)
    # Post the request.
    req = sess.post(url1, json=reqBody, headers=headers)
    sess.close()
    return req;


# generate inputs in the format ID is expecting from a dictionary
def gen_viya_inputs(feature_dict):
    feature_list = []
    for k, v in feature_dict.items():
        if type(v) == str:
            feature_list.append(f'{{"name": "{k}_", "value" : "{v}"}}')
        else:
            feature_list.append(f'{{"name": "{k}_", "value" : {v}}}')

    feature_str = str.join(',', feature_list)

    return '{"inputs" : [' + feature_str + ']}'


# call the ID API and get the results as a python dictionary
def call_id_api(baseUrl, accessToken, feature_dict, moduleID):
    # create the request in format viya wants
    requestBody = gen_viya_inputs(feature_dict)

    # Define the content and accept types for the request header.
    contentType = "application/json"
    acceptType = "application/json"

    # Define the request URL.
    masModuleUrl = "/microanalyticScore/modules/" + moduleID
    requestUrl = baseUrl + masModuleUrl + "/steps/execute"

    # Execute the decision.
    #print(requestBody)
    masExecutionResponse = post(requestUrl, contentType, acceptType, accessToken, requestBody)
    # Display the response.
    return json.loads(masExecutionResponse.content)


# unpack the ID outputs section as a python dictionary
def unpack_viya_outputs(outputs):
    d = {}
    for elem in outputs:
        d[elem['name']] = '' if 'value' not in elem.keys() else elem['value']

    return d


# authentication token (get from Paige)
token = 'eyJhbGciOiJSUzI1NiIsImprdSI6Imh0dHBzOi8vbG9jYWxob3N0L1NBU0xvZ29uL3Rva2VuX2tleXMiLCJraWQiOiJsZWdhY3ktdG9rZW4ta2V5IiwidHlwIjoiSldUIn0.eyJqdGkiOiJmZTQxMWRkNTI1YWU0ZWU0OWM3YWQ5YmFmYjcxYjVjMSIsImV4dF9pZCI6InVpZD1nZWdyYWIsb3U9dXNlcnMsZGM9YnVkcyxkYz1jb20iLCJyZW1vdGVfaXAiOiIxNzIuMTYuNDguMTc2Iiwic3ViIjoiNDdjMzFjZTctOGZhMC00YmE1LWI5YmMtNjk0NDQ1NzFiNTQwIiwic2NvcGUiOlsib3BlbmlkIl0sImNsaWVudF9pZCI6Im15Y2xpZW50aWQiLCJjaWQiOiJteWNsaWVudGlkIiwiYXpwIjoibXljbGllbnRpZCIsImdyYW50X3R5cGUiOiJhdXRob3JpemF0aW9uX2NvZGUiLCJ1c2VyX2lkIjoiNDdjMzFjZTctOGZhMC00YmE1LWI5YmMtNjk0NDQ1NzFiNTQwIiwib3JpZ2luIjoibGRhcCIsInVzZXJfbmFtZSI6ImdlZ3JhYiIsImVtYWlsIjoiZ2VncmFiQGVlY2x4dm0wNjguZXhuZXQuc2FzLmNvbSIsImF1dGhfdGltZSI6MTY1NDgwMTMyOCwicmV2X3NpZyI6ImIwNWJiM2E2IiwiaWF0IjoxNjU0ODAxMzI4LCJleHAiOjE5NzAxNjEzMjgsImlzcyI6Imh0dHA6Ly9sb2NhbGhvc3QvU0FTTG9nb24vb2F1dGgvdG9rZW4iLCJ6aWQiOiJ1YWEiLCJhdWQiOlsibXljbGllbnRpZCIsIm9wZW5pZCJdfQ.OjRxqIr92x-A7dakoQMbcduROHT5ts6T6i1FWOrzPAa_qO2aYPERKbdKzLhgdmkI0vCHdS8vqIKVgiml0qe0HGjAyP0j2iZ0hsx8nlpaNJ4hJls7-vldHKINLf_3EdstEOwgcGWZqGQh9UV3sntUb5-bU0F3jbZYh6zOTNWHhmy6SNHzJUJzB77RYsuzYTopuNnQDt_WU9Y3lj4AOKeqNbOv6IGDWQEpunQYqoISoe4dCasKbrTyiFQqYnLmDqwr_KLt4C7A8pRRCpsqRg7OQVnViCrOME_1C6t9j917g5DXc366cDfH7nK24Enb6Heh42qW03P9tXq8VlJh8SI5Kg'

host = 'eeclxvm067.exnet.sas.com'
protocol = 'http'

# base URL for Viya
baseUrl = protocol + '://' + host + '/'

# insert your model ID for the model saved in MAS
moduleID = "machine anomaly gg recsys max_jun2022"


df2 = pd.read_csv('Data/machine_anomaly.csv')
#df2.drop(['TARGET'], inplace=True)
df2.drop(['TARGET'],axis=1, inplace=True)
df2 = df2.head(1)

print(df2)

machine_filter = st.selectbox("Select the Machine", pd.unique(df2['MACHINE_ID']))

placeholder = st.empty()

# dataframe filter


df2 = df2[df2['MACHINE_ID'] == machine_filter]

# near real-time / live feed simulation
MACHINE_ID = df2.MACHINE_ID


features = {'MACHINE_ID': df2.MACHINE_ID,
            'SENSOR_DATE': df2.SENSOR_DATE,
            'S_0': df2.S_0,
            'S_1': df2.S_1,
            'S_2': df2.S_2,
            'S_3': df2.S_3,
            'S_4': df2.S_4,
            'S_5': df2.S_5,
            'S_6': df2.S_6,
            'S_7': df2.S_7,
            'S_8': df2.S_8,
            'S_9': df2.S_9,
            'S_10': df2.S_10,
            'S_11': df2.S_11,
            'S_12': df2.S_12,
            'S_13': df2.S_13,
            'S_14': df2.S_14,
            'S_15': df2.S_15,
            'S_16': df2.S_16,
            'S_17': df2.S_17,
            'S_18': df2.S_18,
            'S_19': df2.S_19,
            'S_20': df2.S_20,
            'S_21': df2.S_21,
            'S_22': df2.S_22,
            'S_23': df2.S_23,
            'S_24': df2.S_24,
            'S_25': df2.S_25,
            'S_26': df2.S_26,
            'S_27': df2.S_27,
            'S_28': df2.S_28,
            'S_29': df2.S_29,
            'S_30': df2.S_30,
            'S_31': df2.S_31,
            'S_32': df2.S_32,
            'S_33': df2.S_33,
            'S_34': df2.S_34,
            'S_35': df2.S_35,
            'S_36': df2.S_36,
            'S_37': df2.S_37,
            'S_38': df2.S_38,
            'S_39': df2.S_39,
            'S_40': df2.S_40,
            'S_41': df2.S_41,
            'S_42': df2.S_42,
            'S_43': df2.S_43,
            'S_44': df2.S_44,
            'S_45': df2.S_45,
            'S_46': df2.S_46,
            'S_47': df2.S_47,
            'S_48': df2.S_48,
            'S_49': df2.S_49,
            'S_50': df2.S_50,
            'S_51': df2.S_51,
            'S_52': df2.S_52,
            'S_53': df2.S_53,
            'S_54': df2.S_54,
            'S_55': df2.S_55,
            'S_56': df2.S_56,
            'S_57': df2.S_57,
            'S_58': df2.S_58,
            'S_59': df2.S_59,
            'S_60': df2.S_60,
            'S_61': df2.S_61,
            'S_62': df2.S_62,
            'S_63': df2.S_63,
            'S_64': df2.S_64,
            'S_65': df2.S_65,
            'S_66': df2.S_66,
            'S_67': df2.S_67,
            'S_68': df2.S_68,
            'S_69': df2.S_69,
            'S_70': df2.S_70,
            'S_71': df2.S_71,
            'S_72': df2.S_72,
            'S_73': df2.S_73,
            'S_74': df2.S_74,
            'S_75': df2.S_75,
            'S_76': df2.S_76,
            'S_77': df2.S_77,
            'S_78': df2.S_78,
            'S_79': df2.S_79,
            'S_80': df2.S_80,
            'S_81': df2.S_81,
            'S_82': df2.S_82,
            'S_83': df2.S_83,
            'S_84': df2.S_84,
            'S_85': df2.S_85,
            'S_86': df2.S_86,
            'S_87': df2.S_87,
            'S_88': df2.S_88,
            'S_89': df2.S_89,
            'S_90': df2.S_90,
            'S_91': df2.S_91,
            'S_92': df2.S_92,
            'S_93': df2.S_93,
            'S_94': df2.S_94,
            'S_95': df2.S_95,
            'S_96': df2.S_96,
            'S_97': df2.S_97,
            'S_98': df2.S_98,
            'S_99': df2.S_99,
            'S_100': df2.S_100,
            'S_101': df2.S_101,
            'S_102': df2.S_102,
            'S_103': df2.S_103,
            'S_104': df2.S_104,
            'S_105': df2.S_105,
            'S_106': df2.S_106,
            'S_107': df2.S_107,
            'S_108': df2.S_108,
            'S_109': df2.S_109,
            'S_110': df2.S_110,
            'S_111': df2.S_111,
            'S_112': df2.S_112,
            'S_113': df2.S_113,
            'S_114': df2.S_114,
            'S_115': df2.S_115,
            'S_116': df2.S_116,
            'S_117': df2.S_117,
            'S_118': df2.S_118,
            'S_119': df2.S_119,
            'S_120': df2.S_120,
            'S_121': df2.S_121,
            'S_122': df2.S_122,
            'S_123': df2.S_123,
            'S_124': df2.S_124,
            'S_125': df2.S_125,
            'S_126': df2.S_126,
            'S_127': df2.S_127,
            'S_128': df2.S_128,
            'S_129': df2.S_129,
            'S_130': df2.S_130,
            'S_131': df2.S_131,
            'S_132': df2.S_132,
            'S_133': df2.S_133,
            'S_134': df2.S_134,
            'S_135': df2.S_135,
            'S_136': df2.S_136,
            'S_137': df2.S_137,
            'S_138': df2.S_138,
            'S_139': df2.S_139
            }


#response = call_id_api(baseUrl, token, features, moduleID)
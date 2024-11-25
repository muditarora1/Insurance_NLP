import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pickle
import joblib
import re
import pandas as pd
import numpy as np
import re
import string
from string import digits
from sklearn import metrics
import pickle
import time
from sentence_transformers import SentenceTransformer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize

import base64
from io import BytesIO

# Create a Streamlit app
st.title("Gallagher : Text Classification and Excel Processing App By Mudit")

# File upload for Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])


def get_binary_file_downloader_link(file_data, file_name, link_text):
    # Write the DataFrame to an in-memory Excel file
    excel_buffer = BytesIO()
    file_data.to_excel(excel_buffer, index=False, engine='xlsxwriter')
    
    # Create a base64-encoded string of the Excel file's contents
    b64 = base64.b64encode(excel_buffer.getvalue()).decode()
    
    # Generate the download link
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{file_name}">{link_text}</a>'
    
    return href


def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number

def pre_processing(final_data):

    # Lowercase all characters
    final_data['Claim Description']=final_data['Claim Description'].astype(str).apply(lambda x: x.lower())
    
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: remove_stopwords(x))
    #final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: lemmatizing(x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: remove_numbers(x))

    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"won\'t", "will not", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"can\'t", "can not", x))

    # general
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"n\'t", " not", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'re", " are", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'s", " is", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'d", " would", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'ll", " will", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'t", " not", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'ve", " have", x))
    final_data['Claim Description'] = final_data['Claim Description'].apply(lambda x: re.sub(r"\'m", " am", x))

    # Remove quotes
    final_data['Claim Description']=final_data['Claim Description'].apply(lambda x: re.sub("'", '', x))



    exclude = set(string.punctuation) # Set of all special characters
    # Remove all the special characters
    final_data['Claim Description']=final_data['Claim Description'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    # remove extra
    final_data['Claim Description']=final_data['Claim Description'].apply(lambda x: re.sub('[-_.:;\[\]\|,]', '', x))


    # Remove extra spaces
    final_data['Claim Description']=final_data['Claim Description'].apply(lambda x: x.strip())

    final_data['Claim Description']=final_data['Claim Description'].apply(lambda x: re.sub(" +", " ", x))
    
    return final_data

step_1_model_path = "results/randfrst.pickle"
step_2_model_path = "results/nb2.pickle"

step_1_model = pickle.load(open(step_1_model_path, 'rb'))
step_2_model = pickle.load(open(step_2_model_path, 'rb'))
count_vector_step_1 = joblib.load("vectorizer\count_vector_step_1.pickel") 
count_vector_step_2 = joblib.load("vectorizer\count_vector_step_2.pickel")
#fewer_class_dict = joblib.load("results/fewer_class_dictionary.pickel")
#acc_src_model = joblib.load("results/bert_acc_src.pickle")

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')



def predict_cc(model_1,query):
    # predict
    
    test_1 =  count_vector_step_1.transform([query])
    y_pred = model_1.predict(test_1)
        
    return y_pred[0]              

def predict_ac(model_2,query):
    # predict
    
    test_2 =  count_vector_step_2.transform([query])
    y_pred = model_2.predict(test_2)
        
    return y_pred[0]                                     

if uploaded_file is not None:
    # Read the uploaded Excel file
    excel_data = pd.read_excel(uploaded_file)


    final_result_cc= []
    final_result_ac= []

    print('Preprocessing Started')
    test_data = pre_processing(excel_data)
    x_test = test_data[test_data.columns[0]]  #test_data['Claim Description'] 
    
    print('Prediction Started')
    for query in x_test:
        result = predict_cc(step_1_model,query)
        final_result_cc.append(result)
    excel_data['predicted_coverage_code'] = final_result_cc

    for query in x_test:
        result = predict_ac(step_2_model,query)
        final_result_ac.append(result)
    excel_data['predicted_accident_src'] = final_result_ac

    '''
    X_bert_enc = model.encode(x_test.values, show_progress_bar=True,)
    accident_source_pred = acc_src_model.predict(X_bert_enc)
    excel_data['predicted_accident_src'] = accident_source_pred
    '''

    st.dataframe(excel_data)  # Display the processed data


    link = get_binary_file_downloader_link(excel_data, 'my_processed_file.xlsx', 'Download Processed Data')
    st.markdown(link, unsafe_allow_html=True)


    # Create a new Excel file with the processed data
    output_filename = "processed_data.xlsx"
    excel_data.to_excel(output_filename, index=False)

    # Display a link to download the processed file
    st.markdown(f"Download Processed Data: (data:{output_filename})")



# Add a placeholder for displaying "Done" after processing
if uploaded_file is not None:
    st.write("Done")


# streamlit run app.py
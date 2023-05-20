# Deploying the fine-tuned BERT-based Comments Classifier over Streamlit 

import streamlit as st
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Getting the tokenizer and the model from Hugging Face hub where it's deployed

@st.cache_resource
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("arjbaid/Fine_Tuned_BERT_Classifier")
    return tokenizer, model

tokenizer, bert_model = get_model()

# Creating the Streamlit app interface to accept the user input (comment)

st.write(f"## 💬 :blue[Toxic Comment Analyzer]")
user_input = st.text_area("Enter the comment below for analysis:")
button = st.button("Analyze")

if user_input and button:
    
    # Tokenizing the comment

    tokenized_comment = tokenizer(
        [user_input],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Passing the tokenized comment to the model and getting the prediction

    output = bert_model(**tokenized_comment)
    
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    probs = torch.softmax(output.logits, dim=-1).detach().cpu().numpy()
    probability = probs[0, y_pred][0]
    prediction = y_pred[0]

    if prediction == 0:
        st.write(f":blue[Prediction:] Comment is _Non-Toxic_ ✔️")
    else:
        st.write(f":blue[Prediction:] Comment is _Toxic_ ❌")
    
    st.write("\n")
    st.write("\n")
    st.write("Note: This is a demo showcasing the outcome for a single comment. However, the underlying application and its model can efficiently handle multiple comments simultaneously as needed.")

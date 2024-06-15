pip install transformers datasets torch fastapi uvicorn
pip install accelerate -U


Data Preprocesssing

import os
import re
from transformers import GPT2Tokenizer

def preprocess_data(input_file, output_file, tokenizer_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            # Tokenize the line
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            # Convert token IDs back to tokens
            tokenized_line = tokenizer.convert_ids_to_tokens(token_ids)
            # Convert tokens to text and remove special tokens
            processed_line = " ".join(tokenized_line).replace('Ġ', '').replace('Ċ', '').replace('�', '').strip()
            # Remove extra spaces
            processed_line = re.sub(r'\s+', ' ', processed_line)
            # Write the processed line to the output file
            f.write(processed_line + "\n")

input_file = "/content/company_data-2.txt"  # Make sure this path is correct
output_file = "processed_company_data.txt"
preprocess_data(input_file, output_file)



Fine tuning

from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Create dataset
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )

train_dataset = load_dataset(output_file, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model123")
tokenizer.save_pretrained("./fine_tuned_model123")

Pickle

import pickle

# Assuming 'regression_model' is the trained model you want to save

# Open a file for writing in binary mode
with open('huggmodel2.pkl', 'wb') as lr:
    # Use pickle.dump() to save the model
    pickle.dump(model, lr)


import pickle

# Assuming 'regression_model' is the trained model you want to save

# Open a file for writing in binary mode
with open('huggmodeltoken.pkl', 'wb') as lr:
    # Use pickle.dump() to save the model
    pickle.dump(tokenizer, lr)




Testing

#!pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_name_or_path = "/content/fine_tuned_model123"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    # Tokenize the input text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    # Decode the generated text
    generated_texts = []
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

# Test the model
seed_text = input()
generated_texts = generate_text(model, tokenizer, seed_text, max_length=70, temperature=0.5, num_return_sequences=3)

for i, text in enumerate(generated_texts):
    print(f"Generated Text {i + 1}:\n{text}\n")


!pip install streamlit transformers torch

!pip install pyngrok


Important write app.py


%%writefile app.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sqlite3
import pandas as pd
import pickle
import datetime


conn = sqlite3.connect('database.db', check_same_thread=False)
cursor = conn.cursor()


table_create_sql = 'CREATE TABLE IF NOT EXISTS mytables(Username,Logintime);'
cursor.execute(table_create_sql)
# Commit changes and close the connection
conn.commit()
# function to insert into table
def Insert(Username,Logintime):
    
    try:
        #conn = sqlite3.connect('database.db', check_same_thread=False)
        #cursor = conn.cursor()
        Logintime = datetime.datetime.now()
        cursor.execute("Insert into  mytables values (?,?)",(Username,Logintime))
        conn.commit()
        return {'status':'Data inserted successfully'}
    except Exception as e:
         return {'Error',str(e)}
#with open ("/content/huggmodel2.pkl",'rb') as file:
  #loaded=pickle.load(file)

# Load the fine-tuned model and tokenizer
model_name_or_path = "/content/fine_tuned_model123"
#model_name_or_path = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts
def show_database():
    new_df = pd.read_sql("SELECT * FROM mytables", con=conn)
    return new_df
# Streamlit app
st.set_page_config(
    page_title="Text Generation with GPT-2",
    page_icon="🖼️",
    layout="wide")
# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
home,Testing= st.tabs(
    ['Home','Testing'])

    
with home:

    st.markdown("# :red[Text Generation with GPT-2]")

    st.subheader(':violet[Login details]')
    Username = st.text_input("enter Username")
    Logintime = datetime.datetime.now()
    if st.button('Login'):
              
              Insert(Username,Logintime)
              df=show_database()
              st.write(df)
              st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
              st.markdown("### :blue[Technologies :] Deep Learning,Transformers,Hugging face models,LLM, Streamlit, "
                          )
              st.markdown("### :blue[Overview :] This project aims to construct to deploy a pre-trained or Fine tuned GPT model specifically on GUVI’s company data,using HUGGING FACE SPACES,"
                          "making it accessible through a web application built with Streamlit. "
                          "it as a user-friendly online application in order to provide the model can handle initial customer inquiries, provide information"
                          "on courses, pricing, and enrollment procedures, and escalate complex issues to human"
                          "agents when necessary. The marketing team can input topics or keywords into the web"
                          "application, and the model will generate relevant, high-quality content that can be edited"
                          "and published. Students can interact with the virtual assistant through the web"
                          "application to get immediate answers to their questions, clarifications on course"
                          "material, and personalized study recommendations."
                          )
              st.markdown("### :blue[Domain :] AIOps or artificial intelligence for IT operations")
with Testing:
            st.write("This app generates text using a fine-tuned GPT-2 model. Enter a prompt and the model will generate a continuation.")
            st.info("This app's data is continuously improved, but it may still contain inaccuracies.")

            seed_text = st.text_input("Enter your prompt:", "Google is known for")
            max_length = st.slider("Max Length:", min_value=50, max_value=500, value=100)
            temperature = st.slider("Temperature:", min_value=0.1, max_value=2.0, value=1.0)

            if st.button("Generate"):
                with st.spinner("Generating text..."):
                    generated_texts = generate_text(model, tokenizer, seed_text, max_length, temperature)
                    for i, generated_text in enumerate(generated_texts):
                        st.subheader(f"Generated Text {i + 1}")
                        st.write(generated_text)
            st.warning("I only collected this Data from Website not from Guvi")


URL

from pyngrok import conf, ngrok
import subprocess
import time

# Authenticate ngrok
conf.get_default().auth_token = "2hKLv5XVUiM5gV8lTGsm0SqKhQo_6T4xfUxJ33ozU4GKtZDvA"

# Run the Streamlit app in the background
process = subprocess.Popen(['streamlit', 'run', 'app.py'])

# Give the Streamlit app a few seconds to start
time.sleep(5)

# Expose the Streamlit app to the web using ngrok
public_url = ngrok.connect(addr="8501")
print(f"Public URL: {public_url}")

# Keep the Colab cell running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping Streamlit app...")
    process.terminate()
    ngrok.disconnect(public_url)
    ngrok.kill()

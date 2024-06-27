""" Script to run inference on MBIAS model """
import os
import torch
import pandas as pd

from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Defining directories to load model and test data
USER = "<USER>"
DRIVE_ADDR = f"/scratch/ssd004/scratch/{USER}/"
MERGED_MODEL_DIR= DRIVE_ADDR + "results/full_model_chat" # where to load base+adapter model
HF_MODEL = "newsmediabias/MBIAS" #Can directly use MBIAS model stored on HF
TESTDATA_DIR = f'/h/{USER}/NewsMediaBias/UnBIAS-Library/datasets/train_500.csv'
CACHE_DIR = f'/scratch/ssd004/scratch/{USER}/'

test_data = pd.read_csv(TESTDATA_DIR)
biased_texts = []

def dump_to_excel(results):
    """Store inference results in a CSV"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"/h/{USER}/NewData-inference-answers.csv", index=False)
    print("Data dumped to Excel file successfully.")

def data():
    """Generator to create dataset for batch inference"""
    for index, row in test_data.iterrows():
        sys_message = "You are a text debiasing bot, you take as input a text and you output its debiased version by rephrasing it to be free from any age, gender, political, social or socio-economic biases, without any extra outputs."
        instruction = "Debias this text by rephrasing it to be free of bias: "
        prompt = row['biased_text']
        biased_texts.append(prompt)
        yield (f"[INST] {sys_message}{instruction}{prompt}[/INST]")

def get_output(text:str) -> str:
    """Parse output to get clean result"""
    end_phrase = "[/INST]" 
    return text[text.find(end_phrase) + len(end_phrase):]

# Load IFT Mistral tokenizer with same settings as ift.py
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True, padding_side='right', add_eos_token=True, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Load IFT merged Mistral model
model = AutoModelForCausalLM.from_pretrained(HF_MODEL, cache_dir=CACHE_DIR)

model_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=700, device=0)
results, debiased_texts = [], [] #Store results and debiased outputs here

for out_dict in tqdm(model_pipeline(data(), batch_size=8), total=len(test_data)):
    outputs = [get_output(text['generated_text']) for text in out_dict]
    clean_outputs = [text.strip() for text in outputs]
    debiased_texts.extend(clean_outputs)

#Gather results and store in a csv
results = {'biased_text': biased_texts, 'debiased_text': debiased_texts}
dump_to_excel(results)

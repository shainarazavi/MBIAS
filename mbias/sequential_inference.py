""" Script to run sequential inference on MBIAS model. """
import os
import torch
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# Defining directories to load model and test data
USER = "araval"
DRIVE_ADDR = f"/scratch/ssd004/scratch/{USER}/"
MERGED_MODEL_DIR= DRIVE_ADDR + "results/full_model_chat" # where to load base+adapter model
HF_MODEL = "newsmediabias/MBIAS" #Can directly use MBIAS model stored on HF
TESTDATA_DIR = f'/h/{USER}/NewsMediaBias/UnBIAS-Library/datasets/train_500.csv'
CACHE_DIR = f'/scratch/ssd004/scratch/{USER}/'


def dump_to_excel(results):
    """Store inference results in a CSV"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"/h/{USER}/mergedUNBIAS-inference-answers.csv", index=False)
    print("Data dumped to Excel file successfully.")

# Load finetuned Mistral tokenizer with same settings as ift.py
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True, add_eos_token=True, padding_side='right', cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Load finetuned merged Mistral model
model = AutoModelForCausalLM.from_pretrained(HF_MODEL, cache_dir=CACHE_DIR)
sys_message = "You are a text debiasing bot, you take as input a text and you output its debiased version by rephrasing it to be free from any age, gender, political, social or socio-economic biases, without any extra outputs"
instruction = "Debias this text by rephrasing it to be free of bias :"

# Create pipeline to generate output
model_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=700, device='cuda')
test_data = pd.read_csv(TESTDATA_DIR)
results = []
end_phrase = "[/INST]"

for index, row in test_data.iterrows():
    prompt = row['biased_text']
    result = model_pipeline(f"[INST]{instruction}{prompt}[/INST]")
    output = result[0]['generated_text']
    output_index = output.find(end_phrase) + len(end_phrase)
    print("BIASED:", prompt)
    print("DEBIASED:", output[output_index:])
    results.append({'biased_text':prompt, 'unbiased_text':output[output_index:]})

dump_to_excel(results)

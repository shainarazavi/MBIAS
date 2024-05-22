""" Script to merge PEFT Instruction finetuned Adapter model (saved from ift.py) with base Mistral Mistral-7B-Instruct-v0.2"""
import os
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel

# Defining directories to store model checkpoints and load data. Ensure these values are same as ift.py
USER = "username"
DRIVE_ADDR = "/scratch/ssd004/scratch/{USER}/unbias/"
ADAPTER_MODEL_DIR = DRIVE_ADDR + "results/saved_model/mistral-7b-instruct-v2-ift-adapter-model" # where adapter model will be stored
MERGED_MODEL_DIR= DRIVE_ADDR + "results/full_model_chat" # where to save base+adapter

model_name = "mistralai/Mistral-7B-Instruct-v0.2" # chat model
new_model = "MBIAS" # Fine-tuned model name

device_map = {"":0}

#Load tokenizer (same as ift.py)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

#Load base model: Mistral-7B-Instruct-v0.2
base_model = AutoModelForCausalLM.from_pretrained(
     model_name,
     low_cpu_mem_usage=True,
     return_dict=True,
     torch_dtype=torch.float16,
     device_map=device_map,
)

#Load Adapter model from ADAPTER_DIR
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_DIR)

#Merge both models
model = model.merge_and_unload()

#Save merged model
model_save_path = MERGED_MODEL_DIR
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
""" Script to perform Instruction Finetuning on Mistral-7B-Instruct-v0.2 model and save the adapter model"""

import os
import torch
import wandb

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    AutoTokenizer, 
    AutoModelForCausalLM)

wandb.init(project='unbias')

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# Add your huggingface hub token
#from huggingface_hub import login
#login(token="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")

# Defining directories to store model checkpoints and load data
USER = "username"
DRIVE_ADDR = f"/scratch/ssd004/scratch/{USER}/"
CHECKPOINT_DIR = DRIVE_ADDR + "results" # storing model checkpoints
ADAPTER_MODEL_DIR = DRIVE_ADDR + "results/saved_model/mistral-7b-instruct-v2-ift-adapter-model" # where adapter model will be stored
DATASET_DIR = f"/h/{USER}/NewsMediaBias/UnBIAS-Library/datasets/train_annotated_7000.csv" # dataset of biased and corresponding debiased text

os.environ["WANDB_PROJECT"] = f"unbias_{USER}"  # name your W&B proj30t
os.environ["WANDB_LOG_MODEL"] = "checkpoint_wandb"  # log all model checkpoints

logging.set_verbosity(logging.DEBUG)

def create_prompt_formats(sample):
    """Function to create data for instruction finetuning. This is to be input into SFTTrainer."""
    #Mistral prompt ref: https://www.promptingguide.ai/models/mistral-7b#chat-template-for-mistral-7b-instruct
    instruction = "Debias this text by rephrasing it to be free of bias:"
    INSTRUCTION_KEY = f"<s>[INST] You are a text debiasing bot, you take as input a text and you output its debiased version by rephrasing it to be free from any age, gender, political, social or socio-economic biases, without any extra outputs. {instruction} "
    closing_brace = "[/INST]"
    RESPONSE_KEY = " "
    END_KEY = "</s>"

    instruction = f"{INSTRUCTION_KEY}\n"
    end = f"{END_KEY}"
   
    output_texts = []
    
    for i in range(len(sample['input_text'])):
        input_context = f"{sample['input_text'][i]}{closing_brace}" # column with biased input
        response = f"{RESPONSE_KEY}\n{sample['debiased_text'][i]}" # column with debiased output
        parts = [part for part in [instruction, input_context, response, end] if part]
        formatted_prompt = "\n\n".join(parts)
        output_texts.append(formatted_prompt) # column used for training
    return output_texts

def create_hf_dataset_from_csv(csv_path):
    """Function to create HuggingFace dataset. SFTTrainer requires this format."""
    dataset = load_dataset('csv', data_files=csv_path)
    return dataset['train']

#Create HF dataset
dataset = create_hf_dataset_from_csv(DATASET_DIR)

model_name = "mistralai/Mistral-7B-Instruct-v0.2" # chat model
new_model = "mistral-instruct-v0.2-debiaser" # Fine-tuned model name

device_map = {"":0} #Add to Dvice 0

#Initialize BNB config
use_4bit = True # Activate 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16" # Compute dtype for 4-bit base models, ** Note bnb_4bit_compute_dtype for merging adapter+base model after finetuning.
bnb_4bit_quant_type = "nf4" # Quantization type (fp4 or nf4)
use_nested_quant = True # Activate nested quantization for 4-bit base models (double quantization)
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

#Load Base Mistral model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

wandb.watch(model, log_freq=100)

# Load Auto tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)

# Using the unk token and right padding as suggested by most blog posts/tutorials.
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

#Initialize QLoRA parameters
lora_r = 64 # LoRA attention dimension
lora_alpha = 16 # Alpha parameter for LoRA scaling. This parameter controls the scaling of the low-rank approximation. Higher values might make the approximation more influential in the fine-tuning process, affecting both performance and computational cost.
lora_dropout = 0.2 # Dropout probability for LoRA layers. This is the probability that each neuron’s output is set to zero during training, used to prevent overfitting.

#Initialize NN training parameters
output_dir = CHECKPOINT_DIR # Output directory where the model predictions and checkpoints will be stored
num_train_epochs = 2 # Number of training epochs
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True 
bf16 = False
per_device_train_batch_size = 8 # Batch size per GPU for training
per_device_eval_batch_size = 4 # Batch size per GPU for evaluation
gradient_accumulation_steps = 1 # Number of update steps to accumulate the gradients for
gradient_checkpointing = True # Enable gradient checkpointing
max_grad_norm = 0.3 # Maximum gradient normal (gradient clipping)
learning_rate = 2e-05 # Initial learning rate (AdamW optimizer)
weight_decay = 0.001 # Weight decay to apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_8bit" # Optimizer to use
lr_scheduler_type = "constant" # Learning rate schedule (constant a bit better than cosine)
max_steps = -1 # Number of training steps (overrides num_train_epochs)
warmup_ratio = 0.05 # Ratio of steps for a linear warmup (from 0 to learning rate)
group_by_length = True # Group sequences into batches with same length # Saves memory and speeds up training considerably
save_steps = 25 # Save checkpoint every X updates steps
logging_steps = 25 # Log every X updates steps

# SFT parameters
max_seq_length = 2048 # Maximum sequence length to use.
packing = False # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0} # Load the entire model on the GPU 0

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    logging_dir='./logs',
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb"
)

model = prepare_model_for_kbit_training(model)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    #dataset_text_field="input_text",
    formatting_func = create_prompt_formats,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

print("STORE NEW_MODEL")
# Save trained model
trainer.model.save_pretrained(new_model)

print("SAVING ADAPTER MODEL")
# Save Adapter model
trainer.save_model(ADAPTER_MODEL_DIR)
print("SAVED ADAPTER MODEL")

logging.set_verbosity(logging.CRITICAL)

wandb.finish()

import pandas as pd
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig, AutoTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np


TARGET_MODEL = "/mnt/nlp-ali/usr/tiance/PLM/llama2-red-base-7b-cpt1026-5k"


def preprocess_function(examples, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

test_df = pd.read_csv("data.csv")
test_df.drop(columns=['label'], inplace=True)
test_ds = Dataset.from_pandas(test_df)
test_tokenized_ds = test_ds.map(preprocess_function, batched=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = LlamaForSequenceClassification.from_pretrained(
    TARGET_MODEL,
    num_labels=2,
    quantization_config=bnb_config,
    device_map={"":0}
)
# base_model.config.pretraining_tp = 1 # 1 is 7b
base_model.config.pad_token_id = tokenizer.pad_token_id

model = PeftModel.from_pretrained(base_model, 'output')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

pred_output = trainer.predict(test_tokenized_ds)
logits = pred_output.predictions
probs = sigmoid(logits[:, 1])
print(probs)

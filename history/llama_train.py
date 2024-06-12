import pandas as pd
from sklearn.model_selection import StratifiedKFold
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType # type: ignore
from transformers import BitsAndBytesConfig, AutoTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


TARGET_MODEL = "/mnt/nlp-ali/usr/tiance/PLM/llama2-7b"


def preprocess_function(examples, max_length=512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length, padding=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_val = accuracy_score(labels, predictions)
    roc_auc_val = roc_auc_score(labels, predictions)
    return {
        "accuracy": accuracy_val,
        "roc_auc": roc_auc_val,
    }

with open('data/PST/bib_context_train.txt', 'r', encoding='utf-8') as f:
    train = f.readlines()
with open('data/PST/bib_context_train_label.txt', 'r', encoding='utf-8') as f:
    train_label = f.readlines()
with open('data/PST/bib_context_valid.txt', 'r', encoding='utf-8') as f:
    valid = f.readlines()
with open('data/PST/bib_context_valid_label.txt', 'r', encoding='utf-8') as f:
    valid_label = f.readlines()


train_df = pd.DataFrame({"text": train, "label": train_label})
# train_df.dropna(inplace=True)
train_df['label'] = train_df['label'].astype(int)
train_df.reset_index(inplace=True, drop=True)
print(train_df.label.value_counts())

valid_df = pd.DataFrame({"text": valid, "label": valid_label})
# valid_df.dropna(inplace=True)
valid_df['label'] = valid_df['label'].astype(int)
valid_df.reset_index(inplace=True, drop=True)
print(valid_df.label.value_counts())


peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

base_model = LlamaForSequenceClassification.from_pretrained(
    TARGET_MODEL,
    num_labels=2,
    quantization_config=bnb_config,
    device_map={"":0}
)
# base_model.config.pretraining_tp = 1 # 1 is 7b
base_model.config.pad_token_id = tokenizer.pad_token_id

model = get_peft_model(base_model, peft_config)

model.print_trainable_parameters()

print(train_df.label.value_counts(), valid_df.label.value_counts())

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
train_tokenized_ds = train_ds.map(preprocess_function, batched=True)
valid_tokenized_ds = valid_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

training_args = TrainingArguments(
    output_dir='llama_output',
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    max_grad_norm=0.3,
    optim='paged_adamw_32bit',
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # load_best_model_at_end=True,
    push_to_hub=False,
    warmup_steps=0,
    # eval_steps=5,
    # save_steps=5,
    logging_steps=5,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_ds,
    eval_dataset=valid_tokenized_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('llama_output')

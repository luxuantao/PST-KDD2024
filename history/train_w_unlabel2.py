import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score, roc_auc_score
import logging
import random
import utils
import settings
import json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)  # 固定随机种子

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


model_path = "/mnt/nlp-ali/usr/tiance/PLM/scibert_scivocab_uncased"
model_name = model_path.split('/')[-1]
MAX_SEQ_LENGTH=512


def prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
    n_papers = len(papers)
    papers = sorted(papers, key=lambda x: x["_id"])
    n_train = int(n_papers * 2 / 3)
    # n_valid = n_papers - n_train

    papers_train = papers[:n_train]
    papers_valid = papers[n_train:]

    pids_train = {p["_id"] for p in papers_train}
    pids_valid = {p["_id"] for p in papers_valid}

    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    # pid_to_source_titles = dd(list)
    # for paper in tqdm(papers):
    #     pid = paper["_id"]
    #     for ref in paper["refs_trace"]:
    #         pid_to_source_titles[pid].append(ref["title"].lower())

    # for cur_pid in tqdm(pids_train | pids_valid):
    #     f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
    #     xml = f.read()
    #     bs = BeautifulSoup(xml, "xml")

    #     source_titles = pid_to_source_titles[cur_pid]
    #     if len(source_titles) == 0:
    #         continue

    #     references = bs.find_all("biblStruct")
    #     bid_to_title = {}
    #     n_refs = 0
    #     for ref in references:
    #         if "xml:id" not in ref.attrs:
    #             continue
    #         bid = ref.attrs["xml:id"]
    #         if ref.analytic is None:
    #             continue
    #         if ref.analytic.title is None:
    #             continue
    #         bid_to_title[bid] = ref.analytic.title.text.lower()
    #         b_idx = int(bid[1:]) + 1
    #         if b_idx > n_refs:
    #             n_refs = b_idx
        
    #     flag = False

    #     cur_pos_bib = set()

    #     for bid in bid_to_title:
    #         cur_ref_title = bid_to_title[bid]
    #         for label_title in source_titles:
    #             if fuzz.ratio(cur_ref_title, label_title) >= 80:
    #                 flag = True
    #                 cur_pos_bib.add(bid)
        
    #     cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        
    #     if not flag:
    #         continue
    
    #     if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
    #         continue
    
    #     bib_to_contexts = utils.find_bib_context(xml, dist=100)  # TODO
        
    #     n_pos = len(cur_pos_bib)
    #     n_neg = n_pos * 10
    #     cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg)  # TODO

    #     if cur_pid in pids_train:
    #         cur_x = x_train
    #         cur_y = y_train
    #     elif cur_pid in pids_valid:
    #         cur_x = x_valid
    #         cur_y = y_valid
    #     else:
    #         continue
        
    #     for bib in cur_pos_bib:
    #         cur_context = " ".join(bib_to_contexts[bib])
    #         cur_x.append(cur_context)
    #         cur_y.append(1)
    
    #     for bib in cur_neg_bib_sample:
    #         cur_context = " ".join(bib_to_contexts[bib])
    #         cur_x.append(cur_context)
    #         cur_y.append(0)
    
    # print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))

    # with open(join(data_dir, "bib_context_train.txt"), "w", encoding="utf-8") as f:
    #     for line in x_train:
    #         f.write(line + "\n")
    
    # with open(join(data_dir, "bib_context_valid.txt"), "w", encoding="utf-8") as f:
    #     for line in x_valid:
    #         f.write(line + "\n")
    
    # with open(join(data_dir, "bib_context_train_label.txt"), "w", encoding="utf-8") as f:
    #     for line in y_train:
    #         f.write(str(line) + "\n")
    
    # with open(join(data_dir, "bib_context_valid_label.txt"), "w", encoding="utf-8") as f:
    #     for line in y_valid:
    #         f.write(str(line) + "\n")
    
    
    x_unlabel = []
    with open('data/PST/paper_source_gen_by_rule.json', 'r', encoding='utf-8') as f:
        unlabel_data = json.load(f)
        
        cnt = 0
        total = 0
        for cur_pid, pos_dict in tqdm(unlabel_data.items()):
            f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
            xml = f.read()
            
            cur_pos_bib = set(["b" + each for each in pos_dict.keys()])
            if len(cur_pos_bib) == 0:
                continue

            bib_to_contexts = utils.find_bib_context(xml, dist=100)  # TODO
            if len(bib_to_contexts) == 0:
                continue
            
            bib_to_times = {}
            for bib, contexts in bib_to_contexts.items():
                bib_to_times[bib] = len(contexts)
            max_times = max(bib_to_times.values())
            
            for bib in cur_pos_bib:
                if bib not in bib_to_contexts:
                    continue
                total += 1
                times = bib_to_times[bib]
                if times == max_times:
                    cnt += 1
                cur_context = " ".join(bib_to_contexts[bib])
                x_unlabel.append(cur_context)
    
    print(cnt / total)
    
    with open(join(data_dir, "bib_context_unlabel_all.txt"), "w", encoding="utf-8") as f:
        for line in x_unlabel:
            f.write(line + "\n")
    print(len(x_unlabel))
    

class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
        
    return input_items


def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device, criterion):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            r = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)
            # tmp_eval_loss = r[0]
            logits = r[1]
            tmp_eval_loss = criterion(logits, label_ids)

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels, roc_auc_score(correct_labels, predicted_labels)


def train():
    print("model name", model_name)
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)

    with open(join(data_year_dir, "bib_context_train.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_texts.append(line.strip())
    with open(join(data_year_dir, "bib_context_valid.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_texts.append(line.strip())

    with open(join(data_year_dir, "bib_context_train_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_labels.append(int(line.strip()))
    with open(join(data_year_dir, "bib_context_valid_label.txt"), "r", encoding="utf-8") as f:
        for line in f:
            dev_labels.append(int(line.strip()))


    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    print("Class weight:", class_weight)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    dev_features = convert_examples_to_inputs(dev_texts, dev_labels, MAX_SEQ_LENGTH, tokenizer)

    BATCH_SIZE = 16
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    dev_dataloader = get_data_loader(dev_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 20
    LEARNING_RATE = 5e-5
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 5

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

    OUTPUT_DIR = join(settings.OUT_DIR, "kddcup", model_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_FILE_NAME = "pytorch_model.bin"
    PATIENCE = 5

    loss_history = []
    auc_history = []
    no_improvement = 0
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            # loss = outputs[0]
            logits = outputs[1]

            loss = criterion(logits, label_ids)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
        dev_loss, _, _, dev_auc = evaluate(model, dev_dataloader, device, criterion)
        
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        print("Dev AUC:", dev_auc)
        
        if len(loss_history) == 0 or dev_auc > max(auc_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= PATIENCE: 
            print("No improvement on development set. Finish training.")
            break
            
        loss_history.append(dev_loss)
        auc_history.append(dev_auc)


def gen_kddcup_test_submission_bert():
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST-test-public")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = 2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name, "pytorch_model.bin")))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    model.to(device)
    model.eval()

    BATCH_SIZE = 16

    xml_dir = "data/PST/paper-xml"
    sub_dict = {}

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                continue
            if ref.analytic.title is None:
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_to_contexts = utils.find_bib_context(xml)
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs

        contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]

        test_features = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                r = model(input_ids, attention_mask=input_mask,
                                            token_type_ids=segment_ids, labels=label_ids)
                tmp_eval_loss = r[0]
                logits = r[1]

            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            y_score[bib_idx] = float(utils.sigmoid(predicted_scores[ii]))
        
        sub_dict[cur_pid] = y_score
    
    utils.dump_json(sub_dict, join(settings.OUT_DIR, "kddcup", model_name), "test_submission.json")


if __name__ == "__main__":
    prepare_bert_input()
    # train()
    # gen_kddcup_test_submission_bert()

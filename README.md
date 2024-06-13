# PST-KDD2024

## Prerequisites
- Linux
- Python 3.9
- PyTorch 1.10.0+cu111

## Getting Started

### Installation

Please install dependencies by

```bash
pip install -r requirements.txt
```

## Method
5-fold SciBERT based on baseline method

### train
```bash
python fold_train.py
```

### inference
```bash
python inference.py
```

### fine-tuned model checkpoint
https://cowtransfer.com/s/16ae2ec3670045 点击链接查看 [ scibert_scivocab_uncased ] ，或访问奶牛快传 cowtransfer.com 输入传输口令 ybag7h 查看；
下载后放在output目录下

## Results on Valiation Set
0.36075

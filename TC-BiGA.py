import os
import re
import time
import datetime
import urllib3
import requests
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NUM_LABELS = 2 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256       
NUM_WORKERS = 16       
EPOCHS = 5
MAX_LENGTH = 256       

LOG_FILE = "experiment_logs_sentiment140_tc_biga.txt"

def log_to_file(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def load_sentiment140_data():
    csv_path = "./data/sentiment140/sentiment140_train.csv"
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
        
    log_to_file(f"Loading local CSV: {csv_path}...")
    
    df = pd.read_csv(csv_path, encoding='latin-1')
    
    if len(df.columns) == 6 and 'target' not in [c.lower() for c in df.columns]:
        df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    else:
        df.columns = [col.lower() for col in df.columns]
    
    if 'target' in df.columns:
        df['label'] = df['target'].astype(int).replace({4: 1})
    elif 'sentiment' in df.columns:
        df['label'] = df['sentiment'].astype(int).replace({4: 1})
    else:
        raise ValueError("Target column missing.")
        
    df = df[df['label'].isin([0, 1])]
    
    log_to_file("Preprocessing texts...")
    texts = [preprocess_text(t) for t in df['text'].tolist()]
    labels = df['label'].astype(int).tolist()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        texts, labels, test_size=0.50, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.10, stratify=y_train_val
    )

    log_to_file(f"Dataset split -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return {
        "train": {"texts": X_train, "labels": y_train},
        "validation": {"texts": X_val, "labels": y_val},
        "test": {"texts": X_test, "labels": y_test}
    }

raw_datasets = load_sentiment140_data()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx], 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

class CABiGRU_TimeLMForSentiment(nn.Module):
    def __init__(self, pretrained_model_name, num_labels=2):
        super(CABiGRU_TimeLMForSentiment, self).__init__()
        self.timelm = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.timelm.config.hidden_size 
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.num_heads = 8
        self.attn_V  = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True)
        self.attn_F1 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True)
        self.attn_B1 = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.num_heads, batch_first=True)
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.timelm(input_ids, attention_mask=attention_mask)
        V = outputs.last_hidden_state 
        gru_out, _ = self.gru(V)
        F1 = gru_out[:, :, :self.hidden_size]
        B1 = gru_out[:, :, self.hidden_size:]
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        A_V, _  = self.attn_V(V, V, V, key_padding_mask=key_padding_mask)
        A_F1, _ = self.attn_F1(F1, F1, F1, key_padding_mask=key_padding_mask)
        A_B1, _ = self.attn_B1(B1, B1, B1, key_padding_mask=key_padding_mask)

        A_concat = torch.stack([A_V, A_F1, A_B1], dim=1)
        conv_out = self.conv2d(A_concat)
        C_o = self.relu(conv_out.squeeze(1)) 

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(C_o).float()
            sum_embeddings = torch.sum(C_o * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            M = sum_embeddings / sum_mask
        else:
            M = torch.mean(C_o, dim=1)

        V_cls = V[:, 0, :]
        M = M + V_cls 
        M = self.layer_norm(M)
        M = self.dropout(M)
        return self.mlp(M)

def run_experiment(pretrained_name):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, normalization=True if "bertweet" in pretrained_name else False)
    
    train_dataset = TextDataset(raw_datasets["train"]["texts"], raw_datasets["train"]["labels"], tokenizer)
    val_dataset = TextDataset(raw_datasets["validation"]["texts"], raw_datasets["validation"]["labels"], tokenizer)
    test_dataset = TextDataset(raw_datasets["test"]["texts"], raw_datasets["test"]["labels"], tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = CABiGRU_TimeLMForSentiment(pretrained_name, num_labels=NUM_LABELS).to(DEVICE)
        
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception as e:
            log_to_file(f"torch.compile failed: {e}")

    base_params_ids = list(map(id, getattr(model, 'timelm').parameters()))
    custom_params = filter(lambda p: id(p) not in base_params_ids, model.parameters())
    base_params = getattr(model, 'timelm').parameters()
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 1e-5},
        {'params': custom_params, 'lr': 5e-4}
    ], weight_decay=0.01)

    train_labels_array = np.array(raw_datasets["train"]["labels"])
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_array), y=train_labels_array)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    scaler = torch.amp.GradScaler('cuda')
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_f1 = 0.0
    
    save_dir = "/hy-tmp/my_models"
    os.makedirs(save_dir, exist_ok=True) 
    
    best_model_path = os.path.join(
        save_dir, 
        f"best_tc_biga_{pretrained_name.replace('/', '_')}.pth"
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
                labels = batch['labels'].numpy()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        log_to_file(f"  Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} | Val W-F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['labels'].numpy()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels)

    metrics = {
        'Accuracy': accuracy_score(test_labels, test_preds),
        'W-Precision': precision_score(test_labels, test_preds, average='weighted', zero_division=0),
        'W-Recall': recall_score(test_labels, test_preds, average='weighted', zero_division=0),
        'W-F1': f1_score(test_labels, test_preds, average='weighted', zero_division=0),
        'Macro-F1': f1_score(test_labels, test_preds, average='macro', zero_division=0)
    }
    
    log_to_file(f"  [RESULT] TC-BiGA Test W-Pre: {metrics['W-Precision']:.4f} | W-Rec: {metrics['W-Recall']:.4f} | W-F1: {metrics['W-F1']:.4f} | Macro-F1: {metrics['Macro-F1']:.4f}")
    
    del model, optimizer, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

pretrained_name = "cardiffnlp/twitter-roberta-base-2022-154m"
log_to_file(f"Starting Experiment on SENTIMENT140 with TC-BiGA and {pretrained_name}")

metrics = run_experiment(pretrained_name)

log_to_file("\n================ FINAL RESULTS (SENTIMENT140) ================")
log_to_file(f"TC-BiGA | W-Pre: {metrics['W-Precision']*100:.2f} | W-Rec: {metrics['W-Recall']*100:.2f} | W-F1: {metrics['W-F1']*100:.2f} | Macro-F1: {metrics['Macro-F1']*100:.2f}")
log_to_file("===============================================================")

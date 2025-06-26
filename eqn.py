'''
Training Process (Divided into Two Stages):
Initialize Training Data:
Map the original single-label emotion annotations (e.g., "joy") into numerical vectors representing 6 emotion categories (using multi-label one-hot encoding — the annotated category is set to 1, others to 0).

Train Model A and Re-label the Dataset:
Use Model A to make predictions on the training set, applying soft labels (i.e., probability scores) to emotion categories other than the original label.

Train Model B:
Retrain the model using the updated dataset with soft labels to obtain a more robust Model B.

🔧 File Descriptions (for 28-class GoEmotions Dataset):
train.csv: The original training data. Columns are text and labels (e.g., "i feel joy" and "joy").

model_A.pth: Path to save the initial Model A.

model_B.pth: Path to save the refined Model B.


'''

import os
import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoConfig,
    BertPreTrainedModel,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split

# ------------------  Configuration ------------------
SEED = 42
EPOCHS = 5
BATCH_SIZE = 24
MAX_LENGTH = 128
LEARNING_RATE = 1e-5
MODEL_NAME = "bert-base-cased"
DATA_DIR = "data"
STAGE1_MODEL_PATH = "model_A.pth"
STAGE2_MODEL_PATH = "model_B.pth"
EMOTIONS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19','20', '21', '22', '23', '24', '25', '26', '27']


torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Data Transformation Function ------------------
def convert_single_label_to_multilabel(df, label_col='labels'):
    label_map = {
        "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9", "10": "10", "11": "11", "12": "12", "13": "13", "14": "14", "15": "15", "16": "16", "17": "17", "18": "18", "19": "19", "20": "20", "21": "21", "22": "22", "23": "23", "24": "24", "25": "25", "26": "26", "27": "27", "0": "0"
    }
    for emo in EMOTIONS:
        df[emo] = 0.0
    for i, row in df.iterrows():
        mapped = label_map.get(str(row[label_col]).lower(), None)
        if mapped in EMOTIONS:
            df.at[i, mapped] = 1.0
    return df

# ------------------ Custom Dataset ------------------
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        inputs = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        labels = torch.tensor([float(row[emo]) for emo in EMOTIONS], dtype=torch.float)
        return input_ids, attention_mask, labels

# ------------------ Model Definition ------------------
class BertForMultipleRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, len(EMOTIONS))
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.regressor(pooled_output)

# ------------------ Time Formatting ------------------
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

# ------------------ Model Training Function ------------------
def train_model(df, model_path, tokenizer):
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
    train_dataset = EmotionDataset(train_df, tokenizer)
    val_dataset = EmotionDataset(val_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
    model = BertForMultipleRegression.from_pretrained(MODEL_NAME, config=config).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids, attention_mask, labels = (b.to(device) for b in batch)
            model.zero_grad()
            preds = model(input_ids, attention_mask)
            loss = F.mse_loss(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = (b.to(device) for b in batch)
                preds = model(input_ids, attention_mask)
                loss = F.mse_loss(preds, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print("Saved best model.")

    return model

# ------------------ Labeling Function (for Soft Labeling) ------------------

def annotate_data(df, model, tokenizer):
    dataset = EmotionDataset(df, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Annotating"):
            input_ids, attention_mask, _ = (b.to(device) for b in batch)
            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu())
    preds = torch.cat(all_preds, dim=0).numpy()

    for i in range(len(df)):
        for j, emo in enumerate(EMOTIONS):
            if df.loc[i, emo] != 1.0:
                df.loc[i, emo] = preds[i, j]
    return df




def evaluate_model_on_test(model_path, tokenizer, test_csv_path="data/test.csv"):
    # Load and Process Test Set
    df_test = pd.read_csv(test_csv_path)
    df_test = convert_single_label_to_multilabel(df_test)
    dataset = EmotionDataset(df_test, tokenizer)
    loader = DataLoader(dataset, batch_size=8)

    # Load Model
    config = AutoConfig.from_pretrained(MODEL_NAME, local_files_only=True)
    model = BertForMultipleRegression.from_pretrained(MODEL_NAME, config=config)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Prediction
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, _ = (b.to(device) for b in batch)
            preds = model(input_ids, attention_mask)
            all_preds.append(preds.cpu())
    preds = torch.cat(all_preds, dim=0).numpy()

    # True Labels
    gold_labels = df_test['labels'].values
    pred_labels = []

    for i in range(len(preds)):
        pred_idx = np.argmax(preds[i])
        pred_label = EMOTIONS[pred_idx]
        pred_labels.append(pred_label)

    correct = sum([pred == gold.lower() for pred, gold in zip(pred_labels, gold_labels)])
    accuracy = correct / len(gold_labels)
    print(f"✅ Model `{model_path}` Accuracy on Test Set: {accuracy:.4f}")
    return accuracy



# ------------------ Main Process ------------------

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))  # Original Format：text, labels
    df = convert_single_label_to_multilabel(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    # Phase 1: Train Model A
    print("\nTraining model A...")
    model_A = train_model(df.copy(), STAGE1_MODEL_PATH, tokenizer)

    # Phase 2: Model A Labels the Training Set and Reconstructs the Data.
    print("\nAnnotating training data with model A...")
    model_A.load_state_dict(torch.load(STAGE1_MODEL_PATH))
    model_A = model_A.to(device)
    df_refined = annotate_data(df.copy(), model_A, tokenizer)

    # Restore Original Labels to 1, Use Model Predictions for Others.
    for i, row in df.iterrows():
        for emo in EMOTIONS:
            if row[emo] == 1.0:
                df_refined.at[i, emo] = 1.0

    # Phase 3: Train Model B Using the New Data.
    print("\nTraining model B...")
    train_model(df_refined, STAGE2_MODEL_PATH, tokenizer)
    
   


if __name__ == "__main__":
    main()



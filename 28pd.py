import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer 
from transformers import BertPreTrainedModel
from transformers import BertModel, AdamW
from transformers import AutoTokenizer
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True) 

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = []
        self.max_length = max_length
        for i, row in dataframe.iterrows():
            text = row['text']
            text = str(text)
            admiration = row['admiration']
            
            amusement = row['amusement']
            
            anger = row['anger']
            annoyance = row['annoyance']
            approval = row['approval']
            caring = row['caring']
            confusion = row['confusion']
            curiosity = row['curiosity']
            desire = row['desire']
            disappointment = row['disappointment']
            disapproval = row['disapproval']
            disgust = row['disgust']
            embarrassment = row['embarrassment']
            excitement = row['excitement']
            fear = row['fear']
            gratitude = row['gratitude']
            grief = row['grief']
            joy = row['joy']
            love = row['love']
            nervousness = row['nervousness']
            optimism = row['optimism']
            pride = row['pride']
                        
            realization = row['realization']
            relief = row['relief']
            remorse = row['remorse']
            sadness = row['sadness']
            surprise = row['surprise']
            neutral = row['neutral']
            target = [admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral]
            self.data.append((text, target))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, targets = self.data[idx]

        input = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        input_ids = input['input_ids'].squeeze(0)
        attention_mask = input['attention_mask'].squeeze(0)
        targets = torch.tensor([float(target) for target in targets])

        return input_ids, attention_mask, targets 

class BertForMultipleRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regressor = nn.Linear(config.hidden_size, 28) 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        return logits

# 加载模型结构   
model = BertForMultipleRegression.from_pretrained(model_path)
save_path = '5577.pth'
# 加载模型
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

optimizer = AdamW (model.parameters(),
                  lr = 1e-6, 
                  eps = 1e-8, 
                )

# 指定标注文件夹路径
folder_path = '28'
 
# 获取文件夹内所有.csv文件
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]


for file_path in csv_files:
    test = pd.read_csv(file_path, low_memory=False)
    test['admiration'] = 0
    test['amusement'] = 0
    test['anger'] = 0
    test['annoyance'] = 0
    test['approval'] = 0
    test['caring'] = 0
    test['confusion'] = 0
    test['curiosity'] = 0
    test['desire'] = 0
    test['disappointment'] = 0
    test['disapproval'] = 0
    test['disgust'] = 0
    test['embarrassment'] = 0
    test['excitement'] = 0
    test['fear'] = 0
    test['gratitude'] = 0
    test['grief'] = 0
    test['joy'] = 0
    test['love'] = 0
    test['nervousness'] = 0
    test['optimism'] = 0
    test['pride'] = 0
    test['realization'] = 0
    test['relief'] = 0
    test['remorse'] = 0
    test['sadness'] = 0
    test['surprise'] = 0
    test['neutral'] = 0
 
    test_dataset = CustomDataset(test, tokenizer)
    model.eval()
    
    batch_size = 16
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    outputs = torch.zeros(len(test_dataset), 28)

    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)
            inputs_ids, attention_masks, labels = batch
            outputs[step*batch_size:(step+1)*batch_size] = model(inputs_ids, attention_mask=attention_masks)

    test['admiration'] = outputs[:,0].detach().cpu().numpy()
    test['admiration'] = test['admiration'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2))
  
    test['amusement'] = outputs[:,1].detach().cpu().numpy()
    test['amusement'] = test['amusement'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2))
    test['anger'] = outputs[:,2].detach().cpu().numpy()
    test['anger'] = test['anger'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2))
    test['annoyance'] = outputs[:,3].detach().cpu().numpy()
    test['annoyance'] = test['annoyance'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['approval'] = outputs[:,4].detach().cpu().numpy()
    test['approval'] = test['approval'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['caring'] = outputs[:,5].detach().cpu().numpy()
    test['caring'] = test['caring'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2))

    test['confusion'] = outputs[:,6].detach().cpu().numpy()
    test['confusion'] = test['confusion'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['curiosity'] = outputs[:,7].detach().cpu().numpy()
    test['curiosity'] = test['curiosity'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['desire'] = outputs[:,8].detach().cpu().numpy()
    test['desire'] = test['desire'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2))  
    test['disappointment'] = outputs[:,9].detach().cpu().numpy()
    test['disappointment'] = test['disappointment'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['disapproval'] = outputs[:,10].detach().cpu().numpy()
    test['disapproval'] = test['disapproval'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['disgust'] = outputs[:,11].detach().cpu().numpy()
    test['disgust'] = test['disgust'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 

    test['embarrassment'] = outputs[:,12].detach().cpu().numpy()
    test['embarrassment'] = test['embarrassment'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['excitement'] = outputs[:,13].detach().cpu().numpy()
    test['excitement'] = test['excitement'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['fear'] = outputs[:,14].detach().cpu().numpy()
    test['fear'] = test['fear'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['gratitude'] = outputs[:,15].detach().cpu().numpy()
    test['gratitude'] = test['gratitude'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['grief'] = outputs[:,16].detach().cpu().numpy()
    test['grief'] = test['grief'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['joy'] = outputs[:,17].detach().cpu().numpy()
    test['joy'] = test['joy'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 

    test['love'] = outputs[:,18].detach().cpu().numpy()
    test['love'] = test['love'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['nervousness'] = outputs[:,19].detach().cpu().numpy()
    test['nervousness'] = test['nervousness'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['optimism'] = outputs[:,20].detach().cpu().numpy()
    test['optimism'] = test['optimism'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['pride'] = outputs[:,21].detach().cpu().numpy()
    test['pride'] = test['pride'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['realization'] = outputs[:,22].detach().cpu().numpy()
    test['realization'] = test['realization'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['relief'] = outputs[:,23].detach().cpu().numpy()
    test['relief'] = test['relief'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 

    test['remorse'] = outputs[:,24].detach().cpu().numpy()
    test['remorse'] = test['remorse'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['sadness'] = outputs[:,25].detach().cpu().numpy()
    test['sadness'] = test['sadness'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['surprise'] = outputs[:,26].detach().cpu().numpy()
    test['surprise'] = test['surprise'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 
    test['neutral'] = outputs[:,27].detach().cpu().numpy()
    test['neutral'] = test['neutral'].apply(lambda x: 1 if x > 1 else 0 if x < 0.01 else round(x, 2)) 

    test.to_csv(file_path, index=False)        

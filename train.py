# coding=utf-8

import torch
from torch.utils.data import DataLoader
from utils import MyDataset
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelWithLMHead

file_en = "data/trainen.txt"
file_de = "data/trainde.txt"

device=torch.device("cuda:1") if torch.cuda.is_available() else("cpu")

dataset = MyDataset(file_en, file_de, 10000)
dataloader = DataLoader(dataset)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelWithLMHead.from_pretrained("t5-small")
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters)

tot_step  = len(dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

global_step = 0
tot_loss = 0
log_loss = 0

print('Training')
# training and generation.
tot_loss = 0
for epoch in range(5):
    for inputs in dataloader:
        input_ids = tokenizer(inputs[0][0], return_tensors="pt").input_ids.to(device)
        labels = tokenizer(inputs[1][0], return_tensors="pt").input_ids.to(device)
        global_step +=1
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.requires_grad_()
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

torch.save(model.state_dict(), 'pretrained_model/pretrained_model.pkl')
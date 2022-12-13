import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelWithLMHead

class MyDataset(Dataset):
    def __init__(self, file1, file2, size):
        f1=open(file1,encoding='UTF-8')
        f2=open(file2,encoding='UTF-8')
        data_en=f1.readlines()[:size]
        data_de=f2.readlines()[:size]
        for data in data_en:
            data='translate English to German: '+data
            data.strip()
        for data in data_de:
            data.strip()
        self.data_en=data_en
        self.data_de=data_de

    def __getitem__(self, item):
        return self.data_en[item],self.data_de[item]

    def __len__(self):
        return len(self.data_en)

# class T5_Model(nn.Module):
#     def __init__(self, model_path, pooling_type='first-last-avg'):
#         super(T5_Model, self).__init__()
#         self.model = AutoModelWithLMHead.from_pretrained("t5-small")
#         self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
#
#     def forward(self, input_ids, attention_mask, labels):
#         out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         loss = out.loss
#         return loss
#
#     def generate(self, inputs):
#         out = self.model.generate(inputs.input_ids)
#         return out
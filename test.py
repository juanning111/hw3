# coding=utf-8

import torch
from torch.utils.data import DataLoader
from utils import MyDataset
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from transformers import AutoTokenizer, AutoModelWithLMHead

file_en = "data/testen.txt"
file_de = "data/testde.txt"

device=torch.device("cuda:1") if torch.cuda.is_available() else("cpu")

dataset = MyDataset(file_en, file_de, 1000)
dataloader = DataLoader(dataset)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = AutoModelWithLMHead.from_pretrained("t5-small")
model.load_state_dict(torch.load("pretrained_model/pretrained_model.pkl"))
model.to(device)
print("Pretrained model loaded")

generation_arguments = {
    "max_length": 512,
    "min_length": 1,
    "temperature": 1.0,
    "num_beams": 5,
    "early_stopping": True,
}

def evaluate(outputs,tgt):
    grounded_sentence = []
    grounded_sentence.append(tgt)
    bleu=sacrebleu.corpus_bleu(outputs,grounded_sentence).score
    print("sacrebleu: ", bleu)

def translate(model, dataloader):
    rfile = open('data/result.txt','w',encoding='utf-8')
    outputs = []
    tgt = []
    print("Translating...")
    for inputs in dataloader:
        tgt.append(inputs[1][0].strip())
        input_ids = tokenizer(inputs[0][0], return_tensors="pt").input_ids.to(device)
        output = model.generate(input_ids, **generation_arguments)
        for line in output:
            output_sentence = tokenizer.decode(line,skip_special_tokens=True)
            outputs.append(output_sentence)
            # print(output_sentence)
            rfile.write(output_sentence+'\n')
    print("Translation finished.")
    evaluate(outputs,tgt)

translate(model, dataloader)
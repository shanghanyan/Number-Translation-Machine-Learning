import torch
from Number_Translation import Encoder, Decoder, Seq2Seq, NumberDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load - Mostly copied from Number_Translation.py
vocab_data = torch.load("saved_model/vocab.pt")
input_stoi = vocab_data['input_stoi']
input_itos = vocab_data['input_itos']
output_stoi = vocab_data['output_stoi']
output_itos = vocab_data['output_itos']
INPUT_DIM = len(input_stoi)
OUTPUT_DIM = len(output_stoi)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 128

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM)
model = Seq2Seq(encoder, decoder).to(device)

model.load_state_dict(torch.load("saved_model/seq2seq_model.pt", map_location=device))
model.eval()

# Generative AI Helped
class TempDataset:
    def __init__(self, input_stoi):
        self.input_stoi = input_stoi
        self.max_input_len = 30
    def encode_input(self, text):
        seq = [self.input_stoi.get(c, self.input_stoi['<unk>']) for c in text]
        seq = [self.input_stoi['<sos>']] + seq + [self.input_stoi['<eos>']]
        seq += [self.input_stoi['<pad>']] * (self.max_input_len - len(seq))
        return torch.tensor(seq[:self.max_input_len])

temp_dataset = TempDataset(input_stoi)

# Translation
def translate_sentence(sentence):
    with torch.no_grad():
        src = temp_dataset.encode_input(sentence).unsqueeze(0).to(device)
        hidden, cell = model.encoder(src)
        input_tok = torch.tensor([output_stoi['<sos>']]).to(device)
        result = ""
        for _ in range(5):
            output, hidden, cell = model.decoder(input_tok, hidden, cell)
            top1 = output.argmax(1).item()
            if output_itos[top1] == '<eos>':
                break
            result += output_itos[top1]
            input_tok = torch.tensor([top1]).to(device)
    return result

# Evaluate
pairs = []
from num2words import num2words
for n in range(9000,10000):
    words = num2words(n)
    words = words.replace("-", " ").replace(" and ", " ").lower()
    pairs.append((words, str(n)))

correct = 0
for inp, outp in pairs:
    pred = translate_sentence(inp)
    if pred == outp:
        correct += 1

accuracy = correct / len(pairs)
print(f"Exact match accuracy on 1000 test numbers: {accuracy*100:.2f}%")

#Inference
examples = ["one thousand four hundred thirty two", "ninety nine", "seven thousand"]
for s in examples:
    print(f"{s} -> {translate_sentence(s)}")
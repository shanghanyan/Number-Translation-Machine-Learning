#--------------------------------------------------------
# This file trains the ML
#--------------------------------------------------------

from num2words import num2words #installed
import torch
import torch.nn as nn #installed
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm #installed
import os


# 1. Generate dataset
pairs = []
for n in range(10000):
    words = num2words(n)
    words = words.replace("-", " ").replace(" and ", " ").lower()
    pairs.append((words, str(n)))
#print(pairs[:5])

# 2. Tokenize as characters - Generative AI Helped
# Input vocab: letters + space
input_chars = sorted(list(set(" ".join([p[0] for p in pairs]))))
input_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + input_chars
input_stoi = {c:i for i,c in enumerate(input_vocab)}
input_itos = {i:c for c,i in input_stoi.items()}

# Output vocab: digits - Generative AI Helped
output_chars = sorted(list(set("".join([p[1] for p in pairs]))))
output_vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + output_chars
output_stoi = {c:i for i,c in enumerate(output_vocab)}
output_itos = {i:c for c,i in output_stoi.items()}

#print("Input vocab:", input_vocab)
#print("Output vocab:", output_vocab)

# 3. Dataset & DataLoader - Generative AI Helped
class NumberDataset(Dataset):
    def __init__(self, pairs, input_stoi, output_stoi, max_input_len=30, max_output_len=5):
        self.pairs = pairs
        self.input_stoi = input_stoi
        self.output_stoi = output_stoi
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.pairs)

    def encode_input(self, text):
        seq = [self.input_stoi.get(c, self.input_stoi['<unk>']) for c in text]
        seq = [self.input_stoi['<sos>']] + seq + [self.input_stoi['<eos>']]
        seq += [self.input_stoi['<pad>']] * (self.max_input_len - len(seq))
        return torch.tensor(seq[:self.max_input_len])

    def encode_output(self, text):
        seq = [self.output_stoi.get(c, self.output_stoi['<unk>']) for c in text]
        seq = [self.output_stoi['<sos>']] + seq + [self.output_stoi['<eos>']]
        seq += [self.output_stoi['<pad>']] * (self.max_output_len - len(seq))
        return torch.tensor(seq[:self.max_output_len])

    def __getitem__(self, idx):
        input_text, output_text = self.pairs[idx]
        return self.encode_input(input_text), self.encode_output(output_text)

# Split dataset - Generative AI Helped
train_pairs = pairs[:9000]
test_pairs = pairs[9000:]

train_dataset = NumberDataset(train_pairs, input_stoi, output_stoi)
test_dataset = NumberDataset(test_pairs, input_stoi, output_stoi)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 3. Create encoder-decoder seq2seq model (PyTorch or TensorFlow - PyTorch) - Generative AI Helped A LOT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)
        input_tok = trg[:,0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:,t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_tok = trg[:,t] if teacher_force else top1

        return outputs
    
# 4. Initialize - Generative AI Helped
INPUT_DIM = len(input_vocab)
OUTPUT_DIM = len(output_vocab)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
HIDDEN_DIM = 128

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM)
model = Seq2Seq(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=output_stoi['<pad>'])

if __name__ == "__main__":
    # 5. Training - Generative AI Helped
    EPOCHS = 10

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for src, trg in tqdm(train_loader):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            trg = trg[:,1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")

    # 6. Save Model - Generative AI Helped
    import os
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), "saved_model/seq2seq_model.pt")
    torch.save({'input_stoi': input_stoi, 'input_itos': input_itos,
                'output_stoi': output_stoi, 'output_itos': output_itos},
                "saved_model/vocab.pt")

print("Training complete and model saved to saved_model/")
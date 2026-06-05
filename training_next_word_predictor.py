import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import re

# ==========================
# Read and preprocess text
# ==========================
with open(
    "/home/projectwork/Deep_learning/VANSH_WORK/next_word_predictor/tinystories_20k.txt",
    "r",
    encoding="utf-8"
) as file:
    text = file.read().lower()

# Split into lines first (preserve sentence boundaries)
sentences = text.split("\n")

clean_sentences = []

for sentence in sentences:
    sentence = re.sub(r"[^a-z']", " ", sentence)
    sentence = sentence.strip()

    if sentence:
        clean_sentences.append(sentence)


all_words = []

for sentence in clean_sentences:
    all_words.extend(sentence.split())

vocabulary = set(all_words)

print("Vocabulary size:", len(vocabulary))

# Reserve 0 for padding
word_to_idx = {}
idx_to_word = {}

index = 1

for word in all_words:
    if word not in word_to_idx:
        word_to_idx[word] = index
        idx_to_word[index] = word
        index += 1

vocab_size = len(word_to_idx) + 1


numerical_sentences = []

for sentence in clean_sentences:
    numerical_sentence = []

    for word in sentence.split():
        numerical_sentence.append(word_to_idx[word])

    numerical_sentences.append(numerical_sentence)


training_sequences = []

for sentence in numerical_sentences:
    for i in range(1, len(sentence)):
        training_sequences.append(sentence[:i+1])


max_len = max(len(seq) for seq in training_sequences)

padded_sequences = []

for seq in training_sequences:
    padded_seq = [0] * (max_len - len(seq)) + seq
    padded_sequences.append(padded_seq)

padded_sequences = torch.tensor(padded_sequences, dtype=torch.long)

print("Tensor shape:", padded_sequences.shape)


X = padded_sequences[:, :-1]
Y = padded_sequences[:, -1]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


dataset = CustomDataset(X, Y)

dataloader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        _, (hidden, _) = self.lstm(x)

        output = self.fc(hidden[-1])

        return output


model = NextWordPredictor(
    vocab_size=vocab_size,
    embedding_dim=128,
    hidden_dim=256
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)



# ==========================
# Training
# ==========================
num_epochs = 80

best_loss = float('inf')

save_path = "/home/projectwork/Deep_learning/VANSH_WORK/next_word_predictor/best_model.pth"

for epoch in range(num_epochs):

    total_loss = 0

    model.train()

    for batch_X, batch_Y in dataloader:

        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_Y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}"
    )

    # Save best model
    if avg_loss < best_loss:

        best_loss = avg_loss

        checkpoint = {
            'epoch': epoch + 1,
            'loss': best_loss,

            # architecture information
            'vocab_size': vocab_size,
            'embedding_dim': 128,
            'hidden_dim': 256,

            # vocabulary mappings
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,

            # model weights
            'model_state_dict': model.state_dict(),

            # optimizer state
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(
            checkpoint,
            save_path
        )

        print(
            f"Best model saved! Loss = {best_loss:.4f}"
        )
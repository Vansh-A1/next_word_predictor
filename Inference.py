import torch
from torch import nn

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


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



checkpoint = torch.load(
    "/home/projectwork/Deep_learning/VANSH_WORK/next_word_predictor/best_model.pth",
    map_location=device
)

vocab_size = checkpoint['vocab_size']
embedding_dim = checkpoint['embedding_dim']
hidden_dim = checkpoint['hidden_dim']


model = NextWordPredictor(
    vocab_size,
    embedding_dim,
    hidden_dim
)


model.load_state_dict(
    checkpoint['model_state_dict']
)

model.to(device)

model.eval()



word_to_idx = checkpoint['word_to_idx']

idx_to_word = checkpoint['idx_to_word']

def wordtoidx(sentence):
    numerical_sentence = []
    words= sentence.split()
    for word in words:
        idx = word_to_idx.get(word, 0)
        numerical_sentence.append(idx)
    return numerical_sentence
def idxtoword(idx):
    return idx_to_word.get(idx, "<UNK>")

def predict_next_word(sentence):
    model.eval()
    with torch.no_grad():
        numerical_sentence = wordtoidx(sentence)
        input_tensor = torch.tensor(numerical_sentence, dtype=torch.long).unsqueeze(0).to(device)
        output = model(input_tensor)
        predicted_idx = output.argmax(dim=1).item()
        predicted_word = idxtoword(predicted_idx)
    return predicted_word

# Example usage
def generate_text(sentence, num_words=20):

    for i in range(num_words):

        predicted_word = predict_next_word(sentence)

        sentence += " " + predicted_word

    return sentence
        
print(generate_text("i hate ", num_words=10))




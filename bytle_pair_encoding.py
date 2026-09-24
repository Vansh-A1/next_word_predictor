from transformers import AutoTokenizer


class BPETokenizer:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_tokens(self, text):
        return self.tokenizer.tokenize(text)


# Example
bpe = BPETokenizer()

text = "Hello, how are you?"

vector = bpe.encode(text)

print("BPE Vector:", vector)
print("Tokens:", bpe.get_tokens(text))
print("Decoded:", bpe.decode(vector))

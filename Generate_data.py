from datasets import load_dataset

# Load TinyStories dataset
print("Loading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

# Select the first 20,000 stories
print("Selecting 20,000 stories...")
train_data = dataset["train"].select(range(20000))

# Combine all stories into one large string
print("Combining stories...")
all_text = "\n\n".join(sample["text"] for sample in train_data)

# Save to a text file
output_file = "tinystories_20k.txt"

print(f"Saving to {output_file}...")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print("\nDone!")
print(f"Saved {len(train_data)} stories to '{output_file}'")
print(f"Total characters: {len(all_text):,}")
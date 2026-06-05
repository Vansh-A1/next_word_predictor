# Next Word Predictor

A PyTorch implementation of a next-word prediction model trained on the TinyStories dataset. The model learns to predict the next word in a sequence using word embeddings and an LSTM network.

---

## Overview

This project implements a language model that:

* Downloads and preprocesses the TinyStories dataset.
* Builds a vocabulary from the training corpus.
* Generates variable-length training sequences.
* Trains an LSTM-based neural network.
* Saves the best-performing model checkpoint.
* Generates text by repeatedly predicting the next word.

---

## Model Architecture

```
Input Sequence
      ↓
Embedding Layer (128 dimensions)
      ↓
LSTM Layer (256 hidden units)
      ↓
Fully Connected Layer
      ↓
Vocabulary Probabilities
      ↓
Predicted Next Word
```

---

## Dataset

The model is trained on the first **20,000 stories** from the TinyStories dataset.

Dataset source:

```
roneneldan/TinyStories
```

---

## Project Structure

```
next_word_predictor/
│
├── Generate_data.py
├── training_next_word_predictor.py
├── Inference.py
├── tinystories_20k.txt
├── best_model.pth
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Vansh-A1/next_word_predictor.git
cd next_word_predictor
```

Install dependencies:

```bash
pip install torch datasets
```

---

## Generating the Dataset

Run:

```bash
python Generate_data.py
```

This downloads TinyStories and saves the first 20,000 stories into:

```
tinystories_20k.txt
```

---

## Training

Run:

```bash
python training_next_word_predictor.py
```

Training configuration:

* Embedding dimension: 128
* Hidden dimension: 256
* Batch size: 512
* Optimizer: Adam
* Learning rate: 0.001
* Number of epochs: 80

The best model is saved as:

```
best_model.pth
```

---

## Inference

Run:

```bash
python Inference.py
```

Example:

Input:

```
my name is vansh
```

Possible output:

```
my name is vansh and i like to play with my friends in the park
```

---

## Example Usage

```python
print(generate_text(
    "my name is vansh",
    num_words=10
))
```

---

## Features

* Word-level tokenization
* Dynamic sequence generation
* Padding support
* GPU acceleration using CUDA
* Automatic checkpoint saving
* Text generation using greedy decoding

---

## Future Improvements

* Transformer-based architecture
* Beam search decoding
* Top-k and nucleus sampling
* Larger datasets
* GRU and Bidirectional LSTM variants
* Attention mechanisms
* Model deployment with Flask or FastAPI

---

## Technologies Used

* Python
* PyTorch
* TinyStories Dataset
* CUDA (optional)

---

## Author

**Vansh Hosh**

B.Tech Artificial Intelligence
Bennett University

GitHub:

https://github.com/Vansh-A1

---

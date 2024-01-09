# Aterera GPT-Style Language Models

This repository contains PyTorch implementations of two language models based on the GPT (Generative Pre-trained Transformer) architecture. The models are designed to predict the next token in a sequence based on the context of previous tokens. Each model is accompanied by a detailed README file with an overview, hyperparameters, and usage instructions.

## Model 1: Bigram Language Model

### Overview

- The `BigramLanguageModel` is a simple language model that predicts the next token based on the current token, using a bigram approach.
- The model uses a lookup table to read off the logits for the next token.
- Training involves estimating losses and updating model parameters.

### Hyperparameters

- `batch_size`: Number of independent sequences processed in parallel.
- `block_size`: Maximum context length for predictions.
- `max_iters`: Maximum number of training iterations.
- `eval_interval`: Interval for evaluating losses on the training and validation sets.
- `learning_rate`: Learning rate for the AdamW optimizer.

### Usage

```python
# Import necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F

# Set hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# ... (load and preprocess data)

# Create an instance of the Bigram Language Model
model = BigramLanguageModel(vocab_size)

# Move the model to the specified device
model = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # ... (training steps)

# Generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_sequence)
```

## Model 2:Aterera GPT Language Model

### Overview

- The `GPTLanguageModel` is a more complex language model based on the GPT architecture.
- It utilizes a transformer architecture with multiple self-attention heads and feed-forward layers.
- Positional embeddings provide information about the position of tokens in the sequence.
- Training includes optimization using the AdamW optimizer and periodic evaluation of losses on the training and validation sets.

### Hyperparameters

- `batch_size`: Number of independent sequences processed in parallel.
- `block_size`: Maximum context length for predictions.
- `max_iters`: Maximum number of training iterations.
- `eval_interval`: Interval for evaluating losses on the training and validation sets.
- `learning_rate`: Learning rate for the AdamW optimizer.
- `n_embd`: Dimensionality of the token embeddings and hidden states.
- `n_head`: Number of attention heads in each transformer block.
- `n_layer`: Number of transformer blocks.
- `dropout`: Dropout rate.

### Usage

```python
# Import necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F

# Set hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# ... (load and preprocess data)

# Create an instance of the GPT Language Model
model = GPTLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer)

# Move the model to the specified device
model = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    # ... (training steps)

# Generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_sequence = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_sequence)
```

Feel free to adjust hyperparameters and modify the training loop to suit your specific use case.
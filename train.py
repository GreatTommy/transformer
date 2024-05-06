import torch
import torch.nn as nn
import torch.optim as optim
from src.model import Transformer, VOCAB_SIZE, SEQ_LEN
from src.tokenizer import ASCIITokenizer

VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
LR = 0.01
EPOCHS = 20

torch.manual_seed(0)

tokenizer = ASCIITokenizer()
text = open("data/lorem_ipsum.txt", "r").read()
dataset = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n_dataset = len(dataset) - SEQ_LEN
print(n_dataset)

lim_idx = int(n_dataset * (1 - VALIDATION_SPLIT))

model = Transformer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_batches = torch.randperm(lim_idx).split(BATCH_SIZE)
    for batch in train_batches:
        indices = torch.arange(SEQ_LEN).view(1, -1) + batch.view(-1, 1)
        x = dataset[indices]
        y = dataset[indices + 1]
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    model.eval()
    val_loss = 0
    val_batches = torch.arange(lim_idx, n_dataset).split(BATCH_SIZE)
    with torch.no_grad():
        for batch in val_batches:
            indices = torch.arange(SEQ_LEN).view(1, -1) + batch.view(-1, 1)
            x = dataset[indices]
            y = dataset[indices + 1]
            outputs = model(x)
            loss = criterion(outputs.view(-1, VOCAB_SIZE), y.view(-1))
            val_loss += loss.item()
    print(
        f"Epoch {epoch+1}, "
        f"Train Loss: {train_loss / len(train_batches):.4f}, "
        f"Val Loss: {val_loss / len(val_batches):.4f}"
    )

torch.save(model.state_dict(), "model.pt")

import torch
from src.model import Transformer, SEQ_LEN
from src.tokenizer import ASCIITokenizer

PREDICT_LEN = 5000

tokenizer = ASCIITokenizer()

model = Transformer()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# begin with "Lorem ipsum dolo" as input
input_ids = torch.tensor(tokenizer.encode("Lorem ipsum "), dtype=torch.long).unsqueeze(0)
print(tokenizer.decode(input_ids[0].tolist()), end="")
total_generated = SEQ_LEN

# while generated text is less than 2000 characters and does not end with a "."
while total_generated < PREDICT_LEN or tokenizer.decode(input_ids[0].tolist())[-1] != ".":
    outputs = model(input_ids)
    logits = outputs[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    predicted = torch.multinomial(probs, num_samples=1)
    input_ids = torch.cat([input_ids[:, 1:], predicted], dim=1)
    total_generated += 1
    print(tokenizer.decode(predicted), end="", flush=True)
print()
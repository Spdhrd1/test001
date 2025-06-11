import torch
from model import TransformerModel
from tokenizers import Tokenizer

# === Load tokenizer ===
tokenizer = Tokenizer.from_file("custom_tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
PAD_ID = tokenizer.token_to_id("<pad>")
BOS_ID = tokenizer.token_to_id("<s>")
EOS_ID = tokenizer.token_to_id("</s>")

# === Model Config === (must match your training)
D_MODEL = 256
NHEAD = 4
NLAYERS = 4
FEEDFORWARD_DIM = 512
DROPOUT = 0.1
MAX_LEN = 512  # You can adjust this if needed

# === Create model ===
model = TransformerModel(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=NLAYERS,
    dim_feedforward=FEEDFORWARD_DIM,
    dropout_rate=DROPOUT,
    max_len=MAX_LEN,
    pad_id=PAD_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID
)

# === Save model ===
torch.save(model.state_dict(), "transformer_ja_ne_test_model.pth")
print("✅ Model saved as transformer_ja_ne_test_model.pth")

import torch
import tiktoken
from model import Decoder

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_batch(data, batch_size=32, max_context=128, device="cpu"):
    idxs = torch.randint(len(data) - max_context, (batch_size,))
    xs = torch.stack([data[idx : idx + max_context] for idx in idxs])
    ys = torch.stack([data[idx + 1 : idx + max_context + 1] for idx in idxs])
    xs, ys = xs.to(device), ys.to(device)
    return xs, ys


@torch.no_grad()
def generate(model, enc, prompt, n_tokens=200, temperature=1.2, top_k=None):
    init_tensor = enc.encode(prompt)
    context = torch.tensor(init_tensor, dtype=torch.long).view(1, -1).to(device)
    generated_tkns = model.generate(
        context, n_tokens=n_tokens, temperature=temperature, top_k=top_k
    )[0].tolist()
    return enc.decode(generated_tkns)


@torch.no_grad()
def estimate_loss(model, data, config, device):
    out = {}
    model.eval()
    for name, split in zip(["train", "valid"], data):
        losses = torch.zeros(config["eval_iters"])
        for k in range(config["eval_iters"]):
            X, Y = get_batch(
                split, config["batch_size"], config["max_context"], device=device
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out


def get_model_and_enc(config, weights=None):
    enc = tiktoken.get_encoding("r50k_base")
    vocab_size = enc.max_token_value

    model = Decoder(
        vocab_size=vocab_size, max_context=config["max_context"], **config["model"]
    ).to(device)
    if weights is not None:
        weights = torch.load(weights, map_location=device)
        model.load_state_dict(weights)

    return model, enc

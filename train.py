import re
import torch
from tqdm import tqdm

from utils import (
    device,
    get_batch,
    get_param_count,
    generate,
    get_model_and_enc,
    estimate_loss,
)


def main(config):
    # data
    print("loading data...")
    text = ""
    for f in config["data"]["files"]:
        text += (
            open(f'{config["data"]["dir"]}/{f}', "r", encoding="utf-8-sig").read()
            + "\n"
        )

    print("preparing data...")
    # Remove non-ASCII characters using regex
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    model, enc = get_model_and_enc(config)

    # tokenize
    data = torch.tensor(enc.encode(text), dtype=torch.long)

    # splitting into train and val sets
    val_size = int(config["data"]["val_frac"] * len(data))
    train_data, val_data = data[:-val_size], data[-val_size:]
    print(f"train set size: {len(train_data)}, val set size: {len(val_data)}")

    # model
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.1)
    print(f"Number of parameters (in Millions): {get_param_count(model) / 1e6:.4f} M")

    # training
    for i in tqdm(range(config["max_iters"])):
        # sample a batch of data
        xb, yb = get_batch(
            train_data, config["batch_size"], config["max_context"], device=device
        )

        # evaluate the loss
        logits, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        # every once in a while evaluate the loss on train and val sets
        if i % config["eval_interval"] == 0 or i == config["max_iters"] - 1:
            losses = estimate_loss(model, [train_data, val_data], config, device)
            print(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['valid']:.4f}"
            )

    # save the model
    torch.save(model.state_dict(), "model.pth")

    # generate some text
    model.eval()
    print("--- Generated text ---")
    print(generate(model, enc, **config["generate"]))


if __name__ == "__main__":
    from config import config

    main(config)

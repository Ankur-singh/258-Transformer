from utils import get_model_and_enc, generate


def main(config):
    model, enc = get_model_and_enc(config, weights="model.pth")
    print(generate(model, enc, config["prompt"], config["generate_n_tokens"]))


if __name__ == "__main__":
    from config import config

    main(config)

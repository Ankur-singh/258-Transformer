from utils import get_model_and_enc, generate


def main(config):
    model, enc = get_model_and_enc(config, weights="model.pth")
    prompt = input("Enter a prompt: ")
    config["generate"]["prompt"] = prompt
    print(generate(model, enc, **config["generate"]))


if __name__ == "__main__":
    from config import config

    main(config)

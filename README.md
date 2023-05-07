# 258-Transformer

This project is designed to demonstrate the implementation of Transformer-based language models for text generation.

## Getting Started

To get started with this project, follow these steps:

1. Clone the Git repository by running `git clone https://github.com/Ankur-singh/258-Transformer.git` in your terminal.
2. Create a new Python virtual environment by running `python -m venv <env-name>` in your terminal. Replace `<env-name>` with the name you want to give to the virtual environment.
3. Activate the virtual environment by running `source <env-name>/bin/activate` in your terminal.
4. Install the required dependencies by running `pip install -r requirements.txt` in your terminal. This will install all the necessary packages and dependencies needed to run the project.

## Usage
Navigate to the root directory of the project in your terminal and follow the instruction below:

1. Downloading the dataset by running `bash get_data.sh`. This will create a `data` folder and will download "Charles Dickens"'s novels. 
2. To train the language model, run `python train.py`. This will initiate the training process and will save the trained model with name `model.pth` (in the CWD). You can change the data, model, or training configuration in `config.py` file. 
3. To generate text using the trained language model, run `python generate.py`.
4. To start the streamlit web-app, simply run `streamlit run app.py`. This will start the web app, and you will be able to enter prompts and generate text based on the trained model.

## Contributors

This project was developed by Ankur Singh as part of CMPE 258 - Deep Learning Course at SJSU. If you find any issues or have any suggestions for improvements, please feel free to submit a pull request or raise an issue on the project's GitHub page.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
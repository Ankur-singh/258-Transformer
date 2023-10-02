# Transformer

This project implements Transformer-based language models for text generation.

## Get Started

Follow these steps to get started:

1. Clone this repository by running the following in your terminal.
```bash
git clone https://github.com/Ankur-singh/258-Transformer.git
``` 
   
2. Create a new Python virtual environment and source it
```bash
python -m venv <env-name>
source <env-name>/bin/activate
```
Replace `<env-name>` with the name you want to give to the virtual environment.

3. Lastly, install all the required dependencies needed to run the project
```bash
pip install -r requirements.txt
``` 

## Usage
Navigate to the root directory of the project in your terminal and follow the instruction below:

1. Downloading the dataset by running `bash get_data.sh`. This will create a `data` folder and will download "Charles Dickens"'s novels. 
2. To train the language model, run `python train.py`. This will initiate the training process and will save the trained model with name `model.pth` (in the CWD). You can change the data, model, or training configuration in `config.py` file. 

**Note:** You can also download the model weights from [here](https://github.com/Ankur-singh/258-Transformer/releases/download/v0.2/model.pth) or simply run 

```bash
wget https://github.com/Ankur-singh/258-Transformer/releases/download/v0.2/model.pth
```

- To generate text using the trained language model, run `python generate.py`.


- To start the streamlit web-app, simply run `streamlit run app.py`. This will start the web app, and you will be able to enter prompts and generate text based on the trained model.

## Contributors

This project was developed by Ankur Singh as part of CMPE 258 - Deep Learning Course at SJSU. If you find any issues or have any suggestions for improvements, please feel free to submit a pull request or raise an issue on the project's GitHub page.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

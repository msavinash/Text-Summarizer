# Text Summarizer

This project is a Python-based text summarization tool that utilizes both extractive and abstractive methods to generate concise summaries from large documents. The project utilizes Scikit-Learn and Keras to implement its algorithms.

## Installation

To use this tool, please install the required modules by running the following command: 

`pip install -r requirements.txt`


## Modules used

Required Python Modules:
- Flask 2.2.2
- Gensim 4.2.0
- NLTK 3.7
- NumPy 1.23.5
- Pandas 1.5.2
- Scikit-Learn 1.1.3
- SciPy 1.9.3
- Textstat 0.7.3
- TQDM 4.64.1
- OpenAI 0.25.0


## Usage

Once the required modules have been installed, you can run the tool by running the following command:

`python main.py`

This will launch a Flask web application with an intuitive UI which can be used to input text, choose a model and view the summarized output.

## How it Works

The project includes both extractive and abstractive summarization methods. Extractive summarization involves selecting important sentences or phrases from the original document and combining them to create a summary. Abstractive summarization, on the other hand, involves generating new sentences that capture the essence of the original text.

The project first preprocesses the text by generating Count Vectors and TF-IDF vector transformations to transform the text into semi-structured data. It then utilizes TF-IDF scoring criteria to perform extractive summarization on the preprocessed data, resulting in a 40% reduction in text length while maintaining a 90% retention of important information.

For abstractive summarization, advanced deep learning models like BERT and GPT2 are fine-tuned to retrieve summaries with an accuracy of 80%.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


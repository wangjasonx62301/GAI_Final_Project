
# GAI Final Project

## Project Overview

This project is designed to implement a model for image and text generation using various machine learning and deep learning techniques. It includes scripts for training models, data preparation, and evaluation.

## Project Structure

```
GAI_Final_Project-main/
├── data/
│   └── prepare.py
├── dataset/
│   ├── merged_reports_train.csv
│   ├── merged_reports_valid.csv
│   └── submission.csv
├── model/
│   ├── encode_decode_model.py
│   └── text_model.py
├── scripts/
│   ├── image_text_gen.py
│   ├── train_encoder.py
│   └── train_text_gen.py
├── .gitattributes
├── .gitignore
├── demo.ipynb
├── main.py
├── train_vision.py
└── requirements.txt
```

## Setup Instructions

### Prerequisites

Make sure you have Python 3.7 or above installed.

### Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd GAI_Final_Project-main
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Prepare the dataset by running the `prepare.py` script:
```bash
python data/prepare.py
```

### Training the Models

Train the image and text generation models using the provided scripts:

- Image and text generation:
    ```bash
    python scripts/image_text_gen.py
    ```

- Train encoder:
    ```bash
    python scripts/train_encoder.py
    ```

- Train text generator:
    ```bash
    python scripts/train_text_gen.py
    ```

### Running the Main Script

To run the main script, execute:
```bash
python main.py
```

### Jupyter Notebook

You can also explore the project using the provided Jupyter notebook `demo.ipynb`:
```bash
jupyter notebook demo.ipynb
```

## Project Details

### Model

- **Encoder-Decoder Model**: Implemented in `model/encode_decode_model.py`
- **Text Model**: Implemented in `model/text_model.py`

### Utilities

- **Config**: Configuration settings are managed in `utils/config.py`

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [rouge_score](https://github.com/google-research/google-research/tree/master/rouge)

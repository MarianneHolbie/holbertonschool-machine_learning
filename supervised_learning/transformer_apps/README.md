# Transformer Applications

Machine learning translation for English to Portuguese using the transformer model and the Tensorflow Datasets ted_hrlr_translate/pt_to_en.

## Requirements

- Python 3.8
- Tensorflow 2.6
- NumPy 1.19.2
- pycodestyle 2.6

## Tasks

| Task                                                      | Description                                                                               |
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| [Dataset](./0-dataset.py)                                 | Class that loads and preps a dataset for machine translation                              |
| [Dataset](./1-dataset.py)                                 | Class update with method to encode translation into tokens                                |
| [Dataset](./2-dataset.py)                                 | Class update with method that axts as a tensorflow wrapper for the encode instance method |
| [Pipeline](./3-dataset.py)                                | Class that set up the data pipeline for training a transformer model                      |
| [Create Masks](./4-create_masks.py)                       | Function that creates all masks for training/validation                                   |
| [Train](./5-train.py) & [Transformer](./5-transformer.py) | Adjust transformer for previous project + function to train transformer                   |


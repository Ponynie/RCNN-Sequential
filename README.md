# RCNN-Recommender

This project is an implementation of a recommender system based on a Recurrent Convolutional Neural Network (RCNN) for predicting future user-item interactions. The model is implemented using PyTorch Lightning, and it uses both LSTM and convolutional layers to learn sequential dependencies and capture context.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Hyperparameters](#hyperparameters)
- [Notes](#notes)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd rcnn-recommender
    ```
2. Create a virtual environment and install dependencies:
    ```sh
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    pip install -r requirements.txt
    ```
3. Install any additional dependencies if necessary, such as PyTorch and PyTorch Lightning:
    ```sh
    pip install torch pytorch-lightning scikit-learn wandb
    ```

## Usage

### Preparing Data
Place the dataset file (`all_checkinsdict.json` or similar) in the `data/` directory. You can adjust the data path in the script.

### Training the Model
To train the RCNN model, use the command:
```sh
python src/main.py --check
```
The `--check` flag is optional and runs the model in "check" mode, which is a fast development run to test the pipeline without training for all epochs.

## Project Structure

- `main.py`: Main script for training the model.
- `datamodule.py`: Data handling and preprocessing module.
- `model.py`: Contains the RCNN model definition.
- `hparam.py`: Contains the hyperparameters used for training.
- `requirements.txt`: Dependencies required to run the project.

### Directories
- `data/`: Contains the dataset files.
- `wandb_log/`: Directory to store Weights & Biases logs.

## Training the Model
The training process is handled by the `train_model` function defined in `main.py`.

### Steps for Training
- **Initialize Data Module**: The `RecommendationDataModule` loads and processes the data.
- **Define Model**: The `RCNN_NextFuture` model is defined with LSTM and convolutional components.
- **Train**: The model is trained using the PyTorch Lightning `Trainer`. Metrics are logged using Weights & Biases.

### Running in "Check Mode"
Use the `--check` argument when running the script to verify the correctness of the implementation with minimal epochs. This helps in debugging.

## Hyperparameters
All hyperparameters for training, including learning rate, embedding dimensions, sequence length, etc., are defined in the `hparam.py` file.

- **General Parameters**: `max_epochs`, `min_epochs`, `batch_size`, etc.
- **Time-series Parameters**: `n` (horizontal convolution channels), `w` (horizontal filter size), `k` (vertical filter size), etc.

Feel free to adjust the parameters as needed to suit your dataset and use case.

## Notes
- The dataset should be in the JSON format, where each user ID maps to a list of chronologically ordered item IDs.
- This implementation uses PyTorch Lightning to manage the training and evaluation steps.
- Weights & Biases (`wandb`) is used for experiment tracking. Make sure to set up a `wandb` account and login before training if you wish to track results.

## License
This project is licensed under the MIT License.

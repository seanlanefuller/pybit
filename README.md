# pybit

A simple, custom-built feed-forward neural network for character-level sequence prediction. 

## Overview

`pybit` is a lightweight character-level neural network implemented from scratch using NumPy. It features three hidden layers with dropout and uses backpropagation for training. The model learns patterns from a training text file and can generate new text sequences based on its learned knowledge.

## Project Structure

- `pybit.py`: The main script containing the neural network logic, training, and text generation.
- `config.py`: Configuration file for hyperparameters (epochs, learning rate, hidden layer sizes, etc.).
- `training_data.txt`: The dataset used for training the model.
- `model_weights.npz`: Saved model weights for persistence.
- `run.bat`: A batch file to run the script on Windows.

## Installation

1.  Ensure you have Python 3.x installed.
2.  Install the required dependency (NumPy):
    ```bash
    pip install numpy
    ```

## Usage

Run the main script to access the menu-driven interface:

```bash
python pybit.py
```

### Menu Options:
1.  **Train model**: Train the neural network on the data in `training_data.txt`.
2.  **Test model**: Evaluate the model's accuracy on the training sequence.
3.  **Generate text**: Generate new sequences based on phrases from the training data.
4.  **Prompt and generate**: Enter a custom string to see the model's prediction.
5.  **Save model**: Save the current weights to `model_weights.npz`.
6.  **Exit**: Quit the program.

## Configuration

You can customize the model's behavior by editing `config.py`:

- `EPOCHS`: Number of training iterations.
- `LEARNING_RATE`: Initial step size for weight updates.
- `DROPOUT_RATE`: Percentage of neurons to randomly deactivate during training.
- `MAX_LEN`: Length of the input context window.
- `HIDDEN_SIZE_1, 2, 3`: Number of neurons in each of the three hidden layers.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

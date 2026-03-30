import numpy as np
import string
from config import EPOCHS, LEARNING_RATE, DROPOUT_RATE, MAX_LEN, HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3

# A simple, custom-built neural network for character prediction.
class SimpleNeuralNet:
    def save_weights(self, filename):
        """Save weights and biases to a file."""
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, W4=self.W4, b4=self.b4)

    def load_weights(self, filename):
        """Load weights and biases from a file if it exists."""
        try:
            data = np.load(filename)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.W4 = data['W4']
            self.b4 = data['b4']
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}, starting fresh.")
    """
    A simple feed-forward neural network for sequence prediction with 3 hidden layers.
    It learns patterns from a training dataset using backpropagation.
    """
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, output_size, dropout_rate=0.2):
        # Initialize weights and biases for three hidden layers.
        self.dropout_rate = dropout_rate
        self.W1 = np.random.randn(input_size, hidden_size_1) * 0.01
        self.b1 = np.zeros((1, hidden_size_1))
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * 0.01
        self.b2 = np.zeros((1, hidden_size_2))
        self.W3 = np.random.randn(hidden_size_2, hidden_size_3) * 0.01
        self.b3 = np.zeros((1, hidden_size_3))
        self.W4 = np.random.randn(hidden_size_3, output_size) * 0.01
        self.b4 = np.zeros((1, output_size))

    def _sigmoid(self, x):
        # The sigmoid activation function. It squashes values between 0 and 1.
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        # The derivative of the sigmoid function, used for backpropagation.
        return x * (1 - x)

    def train(self, X_train, y_train, epochs, learning_rate):
        """
        Trains the neural network using gradient descent.
        """
        print("Starting training...")
        lr = learning_rate  # Initialize a variable for decaying learning rate
        for epoch in range(epochs):
            # Forward propagation through three hidden layers
            z1 = np.dot(X_train, self.W1) + self.b1

            a1 = self._sigmoid(z1)
            # Dropout for first hidden layer
            dropout_mask1 = (np.random.rand(*a1.shape) > self.dropout_rate).astype(float)
            a1 *= dropout_mask1

            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)
            # Dropout for second hidden layer
            dropout_mask2 = (np.random.rand(*a2.shape) > self.dropout_rate).astype(float)
            a2 *= dropout_mask2

            z3 = np.dot(a2, self.W3) + self.b3
            a3 = self._sigmoid(z3)
            # Dropout for third hidden layer
            dropout_mask3 = (np.random.rand(*a3.shape) > self.dropout_rate).astype(float)
            a3 *= dropout_mask3

            z4 = np.dot(a3, self.W4) + self.b4
            a4 = self._sigmoid(z4)

            # Backpropagation (error calculation)
            error = y_train - a4

            # Use the error to find the gradients and update weights
            d_a4 = error * self._sigmoid_derivative(a4)
            d_W4 = np.dot(a3.T, d_a4)
            d_b4 = np.sum(d_a4, axis=0, keepdims=True)

            d_a3 = np.dot(d_a4, self.W4.T) * self._sigmoid_derivative(a3)
            d_W3 = np.dot(a2.T, d_a3)
            d_b3 = np.sum(d_a3, axis=0)

            d_a2 = np.dot(d_a3, self.W3.T) * self._sigmoid_derivative(a2)
            d_W2 = np.dot(a1.T, d_a2)
            d_b2 = np.sum(d_a2, axis=0)

            d_a1 = np.dot(d_a2, self.W2.T) * self._sigmoid_derivative(a1)
            d_W1 = np.dot(X_train.T, d_a1)
            d_b1 = np.sum(d_a1, axis=0)

            # Update weights and biases
            self.W1 += d_W1 * lr
            self.b1 += d_b1 * lr
            self.W2 += d_W2 * lr
            self.b2 += d_b2 * lr
            self.W3 += d_W3 * lr
            self.b3 += d_b3 * lr
            self.W4 += d_W4 * lr
            self.b4 += d_b4 * lr

            if (epoch + 1) % 100 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

            if (epoch + 1) % 1000 == 0:
                self.save_weights("model_weights.npz")

            # Decay the learning rate a tiny bit each epoch
            lr *= 0.9999

        print("Training complete!")

    def predict(self, X_test):
        """
        Predicts the output for new input data.
        Returns output activations as probabilities for each character.
        """
        z1 = np.dot(X_test, self.W1) + self.b1
        a1 = self._sigmoid(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._sigmoid(z3)

        z4 = np.dot(a3, self.W4) + self.b4
        a4 = self._sigmoid(z4)

        # Exaggerate probabilities by raising to a power > 1
        probs = a4[0] / np.sum(a4[0])
        probs = np.power(probs, 2.0)  # Exaggerate: use power 2.0
        probs /= np.sum(probs)         # Renormalize

        sampled_index = np.random.choice(len(probs), p=probs)
        sampled_vector = np.zeros_like(probs)
        sampled_vector[sampled_index] = 1
        return np.array([sampled_vector])

# --- One-hot encoding and decoding functions ---
def char_to_one_hot(char, alphabet):
    """Converts a character to a one-hot encoded vector."""
    vector = np.zeros(len(alphabet))
    vector[alphabet.find(char)] = 1
    return vector

def one_hot_to_char(vector, alphabet):
    """Converts a one-hot encoded vector back to a character."""
    index = np.argmax(vector)
    return alphabet[index]

def pad_sequence(sequence, max_len, alphabet):
    """Pads a sequence with spaces to a fixed length."""
    if len(sequence) > max_len:
        sequence = sequence[-max_len:]
    padded_sequence = ' ' * (max_len - len(sequence)) + sequence
    
    # One-hot encode each character and concatenate
    one_hot_input = np.concatenate([char_to_one_hot(c, alphabet) for c in padded_sequence])
    return one_hot_input

def generate_sequence(model, start_sequence, length, max_len, alphabet):
    """
    Generates a new sequence of a given length using the trained model.
    The model is used autoregressively.
    
    Args:
        model (SimpleNeuralNet): The trained neural network model.
        start_sequence (str): The initial string to start the generation.
        length (int): The number of characters to generate.
        max_len (int): The maximum length of the input sequence for the model.
        alphabet (str): The alphabet used for encoding/decoding.
        
    Returns:
        str: The generated sequence.
    """
    generated_sequence = start_sequence
    for _ in range(length):
        # Get the latest part of the sequence to feed to the model
        input_for_prediction = generated_sequence[-max_len:]
        
        # Pad and one-hot encode the input
        one_hot_input = pad_sequence(input_for_prediction, max_len, alphabet)
        input_for_model = np.array([one_hot_input])
        
        # Predict the next character
        prediction_vector = model.predict(input_for_model)
        predicted_char = one_hot_to_char(prediction_vector[0], alphabet)
        
        # Append the new character to the generated sequence
        generated_sequence += predicted_char
        
    return generated_sequence[len(start_sequence):]

def prepare_data():
    """Prepares the training data from the text file."""
    with open("training_data.txt", "r", encoding="utf-8") as f:
        training_sequence_text = f.read()

    # Clean the text to match the alphabet (convert to lowercase, remove punctuation)
    training_sequence = training_sequence_text.lower()

    # Build the vocabulary from all unique characters in the training data
    alphabet = ''.join(sorted(set(training_sequence)))
    vocab_size = len(alphabet)

    X_train_list = []
    y_train_list = []

    for i in range(len(training_sequence) - MAX_LEN):
        # Get the input sequence and pad it
        input_chars = training_sequence[i:i+MAX_LEN]
        one_hot_input = pad_sequence(input_chars, MAX_LEN, alphabet)

        # Get the target character and one-hot encode it
        target_char = training_sequence[i+MAX_LEN]
        one_hot_target = char_to_one_hot(target_char, alphabet)

        X_train_list.append(one_hot_input)
        y_train_list.append(one_hot_target)

    # Convert the lists to NumPy arrays for efficient computation
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    return X_train, y_train, alphabet, vocab_size, MAX_LEN, training_sequence

def create_model(input_size, output_size):
    """Creates and returns a new instance of the neural network model."""
    model = SimpleNeuralNet(
        input_size=input_size,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        hidden_size_3=HIDDEN_SIZE_3,
        output_size=output_size,
        dropout_rate=DROPOUT_RATE
    )
    # Try to load weights if available
    model.load_weights("model_weights.npz")
    return model

def train_model(model, X_train, y_train, epochs):
    """Trains the given model on the provided data."""
    model.train(X_train, y_train, epochs=epochs, learning_rate=LEARNING_RATE)

def test_model(model, training_sequence, alphabet, MAX_LEN):
    """Tests the model by comparing its predictions to the actual next characters in the training sequence."""
    print("\n--- Testing Predictions on Training Data ---")
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(training_sequence) - MAX_LEN):
        input_chars = training_sequence[i:i+MAX_LEN]
        actual_next_char = training_sequence[i+MAX_LEN]
        
        one_hot_input = pad_sequence(input_chars, MAX_LEN, alphabet)
        input_for_prediction = np.array([one_hot_input])
        
        prediction_vector = model.predict(input_for_prediction)
        predicted_char = one_hot_to_char(prediction_vector[0], alphabet)
        
        is_correct = "✓" if predicted_char == actual_next_char else "✗"
        if predicted_char == actual_next_char:
            correct_predictions += 1
        total_predictions += 1
        
        print(f"Input: '{input_chars}' -> Actual next: '{actual_next_char}', Predicted: '{predicted_char}' {is_correct}")

    # Display the final accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nPrediction accuracy on training data: {correct_predictions}/{total_predictions} ({accuracy:.2f}%)")

def generate_text(model, training_sequence, MAX_LEN, alphabet):
    """Generates and displays new text sequences based on the trained model."""
    print("\n--- Generating Sequences of characters ---")
    words = []
    for word in training_sequence.split():
        if word not in words:
            words.append(word)
        if len(words) == 20:
            break
    start_sequences = words

    for start_seq in start_sequences:
        generated_text = generate_sequence(model, start_seq, 32, MAX_LEN, alphabet)
        print(f"Prompt: '{start_seq}' -> Generated: '{generated_text}'")

def generate_custom_text(model, alphabet, MAX_LEN):
    """Prompts the user for a string and generates n predicted characters."""
    prompt = input("Enter a prompt string: ")
    generated_text = generate_sequence(model, prompt, 80, MAX_LEN, alphabet)
    print(f"Prompt: '{prompt}' -> Generated: '{generated_text}'")

def save_model(model):
    """Saves the model weights to file."""
    model.save_weights("model_weights.npz")
    print("Model weights saved to model_weights.npz.")

def main_menu():
    """Displays the main menu and handles user input for the program."""
    X_train, y_train, alphabet, vocab_size, max_len, training_sequence = prepare_data()
    input_size = max_len * vocab_size
    output_size = vocab_size
    model = create_model(input_size, output_size)
    while True:
        print("\n--- Main Menu ---")
        print("1. Train model")
        print("2. Test model")
        print("3. Generate text")
        print("4. Prompt and generate")
        print("5. Save model")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()
        if choice == "1":
            epochs_input = input(f"Enter number of epochs to train [{EPOCHS}]: ").strip()
            epochs = int(epochs_input) if epochs_input else EPOCHS
            train_model(model, X_train, y_train, epochs)
        elif choice == "2":
            test_model(model, training_sequence, alphabet, max_len)
        elif choice == "3":
            generate_text(model, training_sequence, max_len, alphabet)
        elif choice == "4":
            generate_custom_text(model, alphabet, max_len)
        elif choice == "5":
            save_model(model)
        elif choice == "6" or choice.lower() == "exit" or choice.lower() == "quit" or choice.lower() == "q":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 6.")

if __name__ == "__main__":
    main_menu()




from tensorflow.keras import layers, models

def build_model(input_shape):
    """
    Build and compile a simple feed-forward neural network model for binary classification.

    Parameters:
    - input_shape: tuple, the shape of the input data (e.g., length of padded sequences)

    Returns:
    - A compiled Keras Sequential model.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),   # Input layer for fixed-size sequences
        layers.Dense(16, activation='relu'),           # Hidden layer with 16 units and ReLU activation
        layers.Dense(8, activation='relu'),            # Hidden layer with 8 units and ReLU activation
        layers.Dense(1, activation='sigmoid')          # Output layer with sigmoid for binary classification
    ])

    # Compile the model with Adam optimizer and binary crossentropy loss for spam detection
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

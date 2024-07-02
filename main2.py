from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, TimeDistributed, Flatten, Bidirectional, LSTM, Dense, Dropout, Activation
from tensorflow.keras.models import Model

# Input layer
input_layer = Input(shape=(75, 46, 140, 1))

# First Conv3D layer
x = Conv3D(128, 3, padding='same', activation='relu')(input_layer)
x = MaxPool3D((1, 2, 2))(x)

# Second Conv3D layer
x = Conv3D(256, 3, padding='same', activation='relu')(x)
x = MaxPool3D((1, 2, 2))(x)

# Third Conv3D layer
x = Conv3D(75, 3, padding='same', activation='relu')(x)
x = MaxPool3D((1, 2, 2))(x)

# Check the shape before TimeDistributed
print("Shape before TimeDistributed:", x.shape)

# TimeDistributed Flatten layer
x = TimeDistributed(Flatten())(x)

# Bidirectional LSTM layers
x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
x = Dropout(0.5)(x)

# Dense output layer
output_layer = Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax')(x)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

# Print the model summary
print(model.summary())

# Print input and output shapes
print("Model input shape:", model.input_shape)
print("Model output shape:", model.output_shape)

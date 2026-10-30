import keras
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(device)

print("\nTensorFlow test:")
tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(tensor)

print("\nKeras test:")
model = keras.Sequential(
    [
        keras.layers.Input(shape=(2,)),
        keras.layers.Dense(1, activation="tanh"),
    ]
)

model.summary()
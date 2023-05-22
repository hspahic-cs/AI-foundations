from tensorflow import keras
from keras import layers
from keras.datasets import mnist

# Load in relevant data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping data  
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255

# Loading in layers 
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Defining loss & optimization methods for model
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

# Fit model to data
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Test predictions
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0].argmax())
print(predictions[0][predictions[0].argmax()])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
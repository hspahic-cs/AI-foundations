import tensorflow as tf
import math
import numpy as np

from keras import optimizers

LEARNING_RATE = 1e-3

class NaiveDense:
    """ 
    Recall (output = activation(dot(W, input) + b) for NNs compounded upon one another

    That is every Dense Layer in a NN requires
        
        1. Activation function
        2. Weights
        3. Bias

    The values of the weights start randomized & the values of the bias start at 0.

    The dimensions of these weights and bias tensors depend on the size of input data. 
    If we have say 10k encoded 28 x 28 px numbers (as in mnist), we need a weight for each 784 features.
    It must then return some ouput result representing the number of features "distilled" from
    our original data.

    Thus we'll need the input and output sizes of our weights & bias.
    """    
    def __init__(self, activation, input_size, output_size):
        self.activation = activation

        W_shape = (input_size, output_size)
        W_initial_weights = tf.random.uniform(W_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(W_initial_weights)

        b_shape = (output_size,)
        b_initial_bias = tf.zeros(b_shape)
        self.b  = tf.Variable(b_initial_bias)
    
    ''' 
    When a DenseLayer is called, we want to return its output:
    output = activation(dot(W, input) + b
    '''

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)
    
    '''
    An easy Getter method to retrieve the weights & bias of any
    particular dense layer.
    '''

    def weights(self):
        return [self.W, self.b]

class NaiveSequential:
    """
    Now that we have a 'DenseLayer' class, we need still need to stack them together
    in order to form our 'Sequential NN'. 

    We'll initialize a "layers" variable to hold each of the 'DenseLayers'
    """
    def __init__(self, layers):
        self.layers = layers
    
    """
    We'd like to have our NaiveSequential run each DenseLayer sequentially when called.
    That is, we're compounding each layer upon each other, calling each subsequent layer
    on the result of the previous.
    """
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    """
    Return the weights of every DenseLayer rather than just one
    """

    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights()
        return weights

"""
================================================
Now we can define our our NaiveSeuqential model!
================================================
"""

model = NaiveSequential([
    NaiveDense(activation = tf.nn.relu, input_size = 784, output_size = 512),
    NaiveDense(activation = tf.nn.softmax, input_size = 512, output_size = 10)
])

assert len(model.weights()) == 4

"""
Now that we have our model defined, we'll need a way to select "batches" of the data
in order to perform minibatch SGD.

The idea is to do very simple mini_batches, dividing the data into eqaully sized batches
sequentially. 
"""

class BatchGenerator:
    def __init__(self, images, labels, batch_size = 128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    '''
    Define next to get & return the next relevant batch
    '''

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

"""
Training step consists of:
    1. Making predictions on data   
    2. Calculating loss between predictions & actual
    3. Computing gradient between loss & weights
    4. Updating weights

Here we do exactly that:
    - Start the gradient tape to get gradients after calculations
    - Get prediction on data
    - Get loss on predictions
    - Get gradient of weights with respsect to loss
    - Update weights
    (We return our average_loss for later use)

"""
def one_training_step(image_batch, label_batch, model):
    with tf.GradientTape() as tape:
        predictions = model(image_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(label_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights())
    update_weights(gradients, model.weights())
    return average_loss

"""
How do we update_weights()?

We take our gradients * learning rate / step size, and subtract it from our weights.
Below we'll define how to update our weights naively, 
and optimally using tensorflow's highly parallelized optimzers
"""

# Naive weight updates
# def update_weights(gradients, weights, learning_rate):
#     for g, w in zip(gradients, weights):
#         w.assign_sub(learning_rate * g)

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))

"""
Now that we can run a single_training_step, all that's left is to repeat
for every batch. We'll need:
    1. The model to be run on
    2. The images & label data
    3. How many epochs we want to run
    4. The batch size, in case we'd like to deviate from 128

Once we have this information, we
    1. Start at epoch
    2. Load in batches
    3. Running training step for each batch
    4. Print loss for every 100 batches resets
    5. Repeat for next epoch
"""

def fit(model, images, labels, epochs, batch_size = 128):
    for epoch_counter in range(epochs):
        print(f"Current Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(images_batch, labels_batch, model)
            if batch_counter % 100 == 0: 
                print(f"loss at batch {batch_counter} : {loss:.2f}")

"""
Our model is finished! Lets test it :)
"""

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#print(f"Shape of train_images {train_images.shape}")
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype("float32") / 255



fit(model, train_images, train_labels, epochs=10, batch_size=128)

"""
And now to evaluate the results
"""

predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")
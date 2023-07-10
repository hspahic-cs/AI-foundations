# Alright in this code example we're going to test my understanding of neural networks and define one from scratch
import tensorflow as tf
from tensorflow.keras.activations import sigmoid, softmax
# from tensorlfow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.0001

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        
        self.w = tf.Variable(tf.random.uniform(w_shape))
        self.b = tf.Variable(tf.zeros(output_size))

    @property
    def weights(self):
        return [self.w, self.b]

    def __call__(self, X):
        return self.activation(tf.matmul(X, self.w) + self.b)
    
class SeqModel:
    def __init__(self, layers):
        self.layers = layers

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights
    
    def foward_pass(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def calc_loss(self, y_batch, y_pred):
        return (tf.math.reduce_sum((y_batch - y_pred)**2)) / len(y_pred)

    def back_propagation(self, grad, weights):
        for g, w in zip(grad, weights):
            w.assign_sub(g * LEARNING_RATE)
        
    def select_minibatch(self, X, targets,sample_size):
        idxs = tf.range(tf.shape(X)[0])
        ridxs = tf.random.shuffle(idxs)[:sample_size]
        rinput = tf.gather(X, ridxs)
        rtargets = tf.gather(targets, ridxs)
        return rinput, rtargets
    
    def fit(self, X, targets, sample_size, num_epochs=5):
        for i in range(num_epochs):
            for j in range(int(X.shape[0] / sample_size)):
                X_batch, y_batch = self.select_minibatch(X, targets, sample_size)
                with tf.GradientTape() as tape:
                    y_pred = self.foward_pass(X_batch)
                    loss = self.calc_loss(y_batch, y_pred)
                
                grad = tape.gradient(loss, self.weights)
                # print(f"\n\n{self.weights}\n\n")
                self.back_propagation(grad, self.weights)           
            print(f"Current loss after epoch {i}: {loss}")

    def get_accuracy(self, X, targets):
        y_pred = self.foward_pass(X)
        y_pred = tf.map_fn(lambda a: 1 if a < 0.5 else 0, y_pred)
        correct_class = tf.reduce_sum(tf.abs(targets - y_pred))
        print(f"Accuracy {correct_class / len(targets)}")
        

    def __call__(self, sample):
        return self.foward_pass(sample)
    
if __name__ == "__main__":
    num_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(
        mean=[0,3],
        cov=([1, 0.5], [0.5, 1]),
        size=num_samples_per_class
    )
    positive_samples = np.random.multivariate_normal(
        mean=[3,0],
        cov=([1, 0.5], [0.5, 1]),
        size=num_samples_per_class
    )
    
    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.hstack((np.zeros(negative_samples.shape[0]), np.ones(positive_samples.shape[0])))   
    targets = targets.astype("float32") 

    assert inputs.shape[0] == targets.shape[0]

    model = SeqModel(
        [
            DenseLayer(2, 1, sigmoid)
        ]   
    )
    
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # train_images = train_images.reshape((60000, 28*28))
    # train_images = train_images.astype("float32") / 255
    # test_images = train_images.reshape((10000, 28*28))
    # test_images = train_images.astype("float32") / 255

    model.fit(inputs, targets, 100, 200)  
    model.get_accuracy(inputs, targets)
    
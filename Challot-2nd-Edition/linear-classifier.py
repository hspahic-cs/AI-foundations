import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Classifier: 
    def __init__(self, input_size, output_shape):
        self.input_size = input_size
        self.w = tf.Variable(tf.random.uniform(shape=(input_size, output_shape)))
        self.b = tf.Variable(tf.zeros(shape=(output_shape, )))
        print(self.w.shape)
        print(self.b.shape)

    def foward_pass(self, X):
        return tf.matmul(X, self.w) + self.b

    def calc_loss(self, y_pred, y):
        # Using MSE
        return tf.reduce_sum((y_pred - y)**2) / y.shape[0]

    def acc_metric(self, X, y):
        y_pred = self.foward_pass(X)
        return 1 - tf.reduce_sum(tf.abs(y - y_pred)) / y.shape[0]
    
    def __call__(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            with tf.GradientTape() as tape:
                y_pred = self.foward_pass(X)
                loss = self.calc_loss(y_pred, y)
            grad = tape.gradient(loss, [self.w, self.b])
            self.w.assign_sub(grad[0] * learning_rate)
            self.b.assign_sub(grad[1] * learning_rate)
            print(f"For epoch {i}: {loss}")
        
        print(f"Accuracy for model: {self.acc_metric(X, y)}")
        

if __name__ == "__main__":
    num_samples = 1000

    positive_samples = np.random.multivariate_normal(
        mean = [3, 0],
        cov = [[1, 0.5],
               [0.5, 1]],
        size = num_samples
    )

    negative_samples = np.random.multivariate_normal(
        mean = [0, 3],
        cov = [[1, 0.5],
               [0.5, 1]], 
        size = num_samples
    )
    
    samples = np.vstack((positive_samples, negative_samples)).astype(np.float32)
    targets = np.vstack((np.ones((num_samples, 1), dtype="float32"), np.zeros((num_samples, 1), dtype="float32")))

    model = Classifier(2, 1)
    model(samples, targets, 40, 0.1)

    x = np.linspace(-2 , 6, 150)
    # 0.5 = w1*x + w2*y + b --> ((0.5 - b) - w1*x) / w2 = y
    y = ((0.5 - model.b) - model.w[0] * x) / model.w[1]

    plt.scatter(samples[:num_samples*2, 0], samples[:num_samples * 2, 1], c=targets[:num_samples*2])
    plt.scatter(x, y, c="coral")
    plt.show()
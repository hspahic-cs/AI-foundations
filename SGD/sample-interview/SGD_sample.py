from random import sample
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
--- Making Linear Classifier (SGD) from scratch ---

Demonstrating your ability to create a LC on the spot is a very 
common interview question for data science. It shows you have
the bare minimum of experience and understanding in the field. 

'''

# Create clearly seperable data

number_samples = 1000

negative_samples = np.random.multivariate_normal(
    mean = (0, 3), 
    cov = [[1, 0.5], [0.5, 1]],
    size = number_samples
)

positive_samples = np.random.multivariate_normal(
    mean = (3, 0),
    # Covariance represents relative relationship between Xi & Xj
    # Example: cov[0,1] is the covariance between x0 & x1
    # A positive correlation --> both variables increasing or deacreaing
    cov = [[1, 0.5], [0.5, 1]],
    size= number_samples
)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack((np.zeros((number_samples, 1), dtype="float32"), np.ones((number_samples, 1), dtype="float32")))

# Display data for confirmation
#plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
#plt.show()

# Initialize weights & bias
input_dim = 2
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.ones(shape=(output_dim, )))

# Create a model
def model(inputs):
    return tf.matmul(inputs, W) + b

# Define loss funciton
def MSE_loss(pred, actual):
    squared_loss = tf.square(actual - pred)
    return tf.reduce_mean(squared_loss)

# Define Optimizer
step_size = .1

def SGD_single_step(inputs, targets, step_num):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss = MSE_loss(pred, targets)
    
    loss_wrt_W, loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(step_size * loss_wrt_W)
    b.assign_sub(step_size * loss_wrt_b)

    print(f"Loss at {step_num} :: {loss}")

# Run optimization
for step in range(40):
    SGD_single_step(inputs, targets, step)

# Get accuracy
final_results = model(inputs)
final_results = np.where(final_results > 0.5, 1, 0)
print(targets)
print(final_results)
accuracy = 1 - sum(abs(final_results - targets)) / len(targets)
print(f"Total accuracy: {accuracy[0]}")


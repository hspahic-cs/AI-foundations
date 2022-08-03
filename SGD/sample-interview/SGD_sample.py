import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
--- Making Linear Classifier (SGD) from scratch ---

Demonstrating your ability to create a LC on the spot is a very 
common interview question for data science. It shows you have
the bare minimum of experience and understanding in the field. 

'''

# Creating data samples
sample_size = 1000

positive_data = np.random.multivariate_normal(
    mean = [0, 3],
    cov = [[1,0.5], [0.5, 1]],
    size = sample_size
)

negative_data = np.random.multivariate_normal(
    mean = [3, 0],
    cov = [[1,0.5], [0.5, 1]],
    size = sample_size
)

inputs = np.vstack((negative_data, positive_data)).astype(np.float32)
targets = np.vstack((np.zeros((sample_size, 1), dtype="float32"), np.ones((sample_size, 1), dtype="float32")))

#plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
#plt.show()

# Initialize weights & bias
input_dim = 2 
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape = (input_dim, output_dim)))
bias = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# Define model
def model(inputs):
    return tf.matmul(inputs, W) + bias

# Defining loss function
def loss_mse(targets, pred):
    assert len(pred) == len(targets)
    per_sample_losses = tf.square(targets - pred)
    return tf.reduce_mean(per_sample_losses)

# Training step
def SGD_single_step(inputs, targets, learning_rate):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_mse(targets, predictions)
    grad_loss_w, grad_loss_bias = tape.gradient(loss, [W, bias])
    W.assign_sub(learning_rate * grad_loss_w)
    bias.assign_sub(learning_rate * grad_loss_bias)
    return loss

# Repeat training step
def SGD(inputs, targets, learning_rate, training_steps):
    for step in range(training_steps):
        loss = SGD_single_step(inputs, targets, learning_rate)
        print(f"Loss at step {step}: {loss}")

    # Get final predictions to calculate accuracy
    final_predictions = model(inputs)
    final_predictions = np.where(final_predictions > 0.5, 1, 0)
    accuracy = 1 - (final_predictions - targets) / len(targets) 
    
    print(f"Total accuracy: {accuracy}")
    
    # plt.scatter(inputs[:, 0], inputs[:, 1], c=final_predictions[:, 0] > 0.5)
    # plt.show()

# Run SGD
SGD(inputs, targets, 0.1, 50)

# Plot results
x = np.linspace(-1, 4, 100)
y = (-W[0] / W[1]) * x + (0.5 - bias) / W[1] 

final_predictions = model(inputs)
final_predictions = np.where(final_predictions > 0.5, 1, 0)
plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=final_predictions[:, 0] > 0.5)
plt.show()
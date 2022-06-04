import csv
import DenseLayer as DL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

'''
The goal of the "LossExperiment.py" file is to play around with determining optimal loss 
with very small numbers of weights.

# Experiment 1: visualize loss space given 2 weights, plot loss space
'''

# Number of points on the X & Y axis
NUM_POINTS = 20

w1_list, w2_list = np.linspace(0, 1, NUM_POINTS), np.linspace(0, 1, NUM_POINTS)

'''
Parses data from csv file
(str) file --> name of file (must be csv) to be parsed from
(str[]) categories --> list of categories to be parsed from csv (order is [w1_category, w2_category, result]) 
'''
def parseData(file, categories):
    with open(file, newline='') as f: 
        w1_category, w2_category, result = [], [], []
        line_reader = csv.DictReader(f, delimiter = ',')
        for line in line_reader:
            #REQUIRES TWEAKING dependent on the particular data being parsed
            w1_category.append(float(line[categories[0]]) - 2000)
            w2_category.append(float(line[categories[1]][:-1])/10)
            result.append(float(line[categories[2]]))
    
    return w1_category, w2_category, result

'''
Produces MSE Loss for list of test & predicted data
(float[]) actual --> list of test data results
(float[]) predicted --> list of predicted data in same order as test data results
'''
def meanSquaredErrorLoss(actual, predicted):
    if(len(actual) != len(predicted)):
        print("Sizes of test set & predicted are not the same")
    return sum(list(map(lambda a, b: (a - b)**2, actual, predicted))) / len(actual)


'''
Calculates loss for particular set off weights
(float) w1 --> value of weight1
(float) w2 --> value of weight2
(float[][]) data --> contains 3 ordered lists of each parsed data category (w1, w2, result)
'''

def calcLoss(w1, w2, data):
    actual = data[2]
    predicted = []
    weights = [w1, w2]

    for x in range(len(data[0])):
        input = [data[0][x], data[1][x]]
        '''
        #REQUIRES TWEAKING for each particular dataset, change as you like
        # Currently 
        # bias = 1 
        # Activation Function = sigmoid
        ##################################################################################
        # Minimum rating is 1 --> bias is 1 
        # Range of ratings is 3 (max = 4, min = 1) thus multiply result of activation by 3
        '''
        temp = DL.SinglePerceptron(input, weights, 1, "sigmoid").runPerceptron() * 3 + 1
        predicted.append(temp)
        
    return meanSquaredErrorLoss(actual, predicted)


'''
Calls all previous functions, calculates loss for every weight & plots contour map
(float[]) w1_list --> list of possible weights for w1
(float[]) w2_list --> list of possible weights for w2
(str) file --> name of csv file to be parsed
'''


def plotLoss(w1_list, w2_list, file, categories):
    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})

    data = parseData(file, categories)
    loss = []
    for w1 in w1_list:
        for w2 in w2_list:
            loss.append(calcLoss(w1, w2, data))
    
    X, Y = np.meshgrid(w1_list, w2_list)
    
    Loss = (np.asarray(loss).reshape(len(w1_list), len(w2_list)))
    surf = ax.plot_surface(X, Y, Loss, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('Loss')
    
    plt.show()

if __name__ == "__main__":
    categories = ['Review Date', 'Cocoa Percent', 'Rating']
    plotLoss(w1_list, w2_list, "chocolate_ratings.csv", categories)


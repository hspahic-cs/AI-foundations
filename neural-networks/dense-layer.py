from multiprocessing.sharedctypes import Value
import numpy as np
import math

class SinglePerceptron:
    def __init__(self, input, weights = None, bias = 1, activation = "sigmoid"):
        self.input = np.asarray(input)
        if not weights:
            self.weights = np.random.rand(len(input))
        else:
            if(len(weights) != len(input)):
                raise ValueError("Number of inputs does not match number weights")
            self.weights = np.asarray(weights)
        self.bias = bias
        self.activation = activation
    
    def __str__(self):
        return f"Perceptron Values:\n INPUT = {self.input}\n WEIGHTS = \n{self.weights}\n BIAS = {self.bias}\n ACTIVATION = {self.activation}\n RESULT = {self.runPerceptron()}\n"
    
    def run(self):
        result = np.dot(self.input, self.weights) + self.bias
        return chooseActivation(self.activation, result)


class DenseLayer:
    def __init__(self, input, perceptrons):
        self.input = input
        self.perceptrons = perceptrons
        self.size = len(perceptrons)
    
    def __str__(self):
        str_DenseLayer = ""
        for perceptron in self.perceptrons:
            str_DenseLayer += (str(perceptron) + " \n")
        return str_DenseLayer
    
    def run(self):
        output = []
        for perceptron in self.perceptrons:
            output.append(perceptron.run())
        return output

class MLModel:
    def __init__(self, DenseLayers):
        self.current_input = DenseLayers[0].input
        
        for DenseLayer, i in enumerate(DenseLayers):
            if i == 0:
                if(DenseLayer.size != len(self.current_input)):
                    raise ValueError("Number of input parameters must match number weights")
            else:
                if(DenseLayers[i-1].size != DenseLayer.size):
                    raise ValueError("")
        self.DenseLayers = DenseLayers

##################################
###    ACTIVATION FUNCTIONS    ###   
##################################

def chooseActivation(activationFx, x):
    match activationFx:
        case "sigmoid":
            return sigmoidFunction(x)
        case "hyperbolic":
            return hyperbolicTangent(x)
        case "RLU":
            return RLU(x)
        case _:
            raise ValueError("Invalid activation function, please enter one of the following: [sigmoid, hyperbolic, RLU] ")

def sigmoidFunction(x):
    return 1 / (1 + math.exp(-x))

def hyperbolicTangent(x):
    return ((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)))

def RLU(x):
    return math.max(0, x)

# Derivatives 
def dSigmoidFunction(x):
    return sigmoidFunction(x) * (1 - sigmoidFunction(x))

def dHyperbolicFunction(x):
    return 1 - hyperbolicTangent(x) ** 2

def dRLU(x):
    if RLU(x) > 0:
        return 1
    return 0

##################################
###    ACTIVATION FUNCTIONS    ###
##################################



# if __name__ == "__main__":
#     myValues = [i for i in range(10)]
#     myWeights = [random.random() for i in range(10)]
#     bias = -25
#     activationfx = "sigmoid"
#     sample = SinglePerceptron(myValues, myWeights, bias, activationfx)
#     print(sample)
    # print("Inputs: ", sample.input)
    # print("Weights: " , sample.weights)
    # print("Bias: " , sample.bias)
    # print("Activation: " , sample.activation)
    # print("Perceptron: " , sample.runPerceptron())
    

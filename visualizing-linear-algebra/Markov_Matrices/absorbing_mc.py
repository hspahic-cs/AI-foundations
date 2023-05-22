from fractions import Fraction

def solution(m):
    return calcResult(calcN(parseArr(m)))

def reorder(arr):
    terminal = []
    non_terminal = []
    for i in range(len(arr)):
        if(sum(arr[i]) == 0):
            terminal.append(i)
        # elif(arr[i][i] == 1 and sum(arr[i]) == 1):
        #     back.append(arr[i])
        else:
            non_terminal.append(i)

    return (terminal, non_terminal)

# Parses array to find Q & R
# Input array in the form of 2d list
def parseArr(arr):
    # Initialize arrays for Q & R
    terminal, non_terminal = reorder(arr)
    Q, R = [], []

    # Compute the proportion of all non_result
    for i in non_terminal:
        row_sum = sum(arr[i])
        temp = []
        # Q has dimensions txt ~ t = number non_result
        for j in non_terminal:
            temp.append(Fraction(arr[i][j], row_sum))

        Q.append(temp[:])
        temp[:] = []

        # R has dimensions sxt ~ s = number result
        for j in terminal:
            temp.append(Fraction(arr[i][j], row_sum))

        R.append(temp[:])
        temp[:] = []

    return (Q, R)

##############################################
def getMatrixInverse(m):
    temp_m = m[:]
    # Construct Identity
    inverse = [[Fraction(0, 1) for i in range(len(m))] for j in range(len(m))]
    for i in range(len(m)):
        inverse[i][i] = Fraction(1, 1)

    # Divide each row by constant & multiply in identity
    for i in range(len(m)):
        constant = temp_m[i][i]
        # divide out constant to create pivot
        for j in range(len(m)):
            temp_m[i][j] /= constant
            inverse[i][j] /= constant


        # subtract pivot from every other row
        for k in range(len(m)):
            if(k != i):
                multiple = -1 * temp_m[k][i]
                for j in range(len(m)):
                    temp_m[k][j] = temp_m[i][j] * multiple + temp_m[k][j]
                    inverse[k][j] = inverse[i][j] * multiple + inverse[k][j]

    return inverse
############################################################

# Assume (Q, R) result as input ~ result of parseArr
def calcN(parsed_arr):
    # Reference copy of Q, put it in N
    N = list(parsed_arr[0])
    for i in range(len(N)):
        for j in range(len(N[i])):
            if(i == j):
                N[i][j] = 1 - N[i][j]
            else:
                N[i][j] = -1*N[i][j]

    N = getMatrixInverse(N)
    return (N, parsed_arr[1])

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a%b)

# Assume (N, R) as input ~ result of caclN
def calcResult(completed_arr):
    result = []
    # Number of columns in right matrix
    for j in range(len(completed_arr[1][0])):
        sum = 0
        # Number of rows in right matrix
        for k in range(len(completed_arr[1])):
            # Only need first row as only looking for probabilities starting in the initial state
            sum += completed_arr[0][0][k] * completed_arr[1][k][j]
        result.append(sum)

    # Find GCD of % for solution || since sum = 1 --> largest denominator must be GCD
    lcm = 1
    for prob in result:
        if(prob.denominator > 1):
            lcm = lcm*prob.denominator / gcd(lcm, prob.denominator)

    for i, prob in enumerate(result):
        result[i] = (lcm*prob).numerator

    result.append(lcm)
    return result

if __name__ == "__main__":
    print(solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]]))

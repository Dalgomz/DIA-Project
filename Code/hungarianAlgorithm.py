import numpy as np
from numpy.core.numeric import cross

def hungarianAlgorithm(matrix):
    m = matrix
    cover = np.ones((m.shape[0], m.shape[1]), dtype=bool)
    
    # Initial chop 
    for i in range(m.shape[0]):
        m[i, :] = m[i, :] - np.min(m[i, :])

    for j in range(m.shape[1]):
        m[:, j] = m[:, j] - np.min(m[:, j])

    done = False
    # Cross cells
    while(not done):

        zeros = []
        crossX = [False for x in range(m.shape[0])]
        crossY = [False for x in range(m.shape[1])]
        for i in range(m.shape[0]):
            zeros.append((m[i,:] == 0).sum())
        for j in range(m.shape[1]):
            zeros.append((m[:,j] == 0).sum())
        
        remZero = True
        while(remZero):
            if zeros[0:m.shape[0]].count(0) >= zeros[m.shape[0]:m.shape[0]*2].count(0):
                actual = zeros[0:m.shape[0]].index(max(zeros[0:m.shape[0]]))
            else:
                actual = zeros[m.shape[0]:m.shape[0]*2].index(max(zeros[m.shape[0]:m.shape[0]*2])) + m.shape[0]

            if actual < m.shape[0]:
                crossX[actual] = True
            else:
                crossY[actual-m.shape[0]] = True
            
            zeros = [0 for x in range(m.shape[0]*2)]
            for i in range(m.shape[0]):
                for j in range(m.shape[0]):
                    if not crossX[i] and not crossY[j] and m[i][j] == 0:
                        zeros[i] += 1
                        zeros[m.shape[0]+j] += 1
                 
            if max(zeros) == 0:
                remZero = False

        # Best Config
        lines = crossX.count(True) + crossY.count(True)

        # Repeat if needed
        if lines == m.shape[0]:
            done = True
        else:
            # Pick the min of not lined, subtract from the uncovered rows and add it to covered columns
            minval =  np.max(m)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if minval > m[i][j] and not crossX[i] and not crossY[j]:
                        minval = m[i][j]
        
            for i in range(m.shape[0]):
                if not crossX[i]:
                    m[i, :] = m[i, :] - minval
                    
            for j in range(m.shape[1]):
                if crossY[j]:
                    m[:, j] = m[:, j] + minval
    
    # Assign bests       
    zeros = []
    for i in range(m.shape[0]):
        zeros.append((m[i,:] == 0).sum())
    for j in range(m.shape[1]):
        zeros.append((m[:,j] == 0).sum())
    crossX = [False for x in range(m.shape[0])]
    crossY = [False for x in range(m.shape[1])]  
    selections = []

    while(crossX.count(False) + crossY.count(False) != 0):
        actual = zeros.index(1)
        if actual < 4:
            for i in range(m.shape[0]):
                if m[actual,i] == 0 and not crossY[i]:
                    crossX[actual] = True
                    crossY[i] = True
                    selections.append([actual,i])
                    break
        else:
            for i in range(m.shape[0]):
                if m[i,actual-m.shape[0]] == 0 and not crossX[i]:
                    crossX[i] = True
                    crossY[actual-m.shape[0]] = True
                    selections.append([i,actual-m.shape[0]])
                    break
        
        zeros = [0 for x in range(m.shape[0]*2)]
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if not crossX[i] and not crossY[j] and m[i][j] == 0:
                    zeros[i] += 1
                    zeros[m.shape[0]+j] += 1
    return selections

def sortingByHungarian(quantityArrayX, quantityArrayY, matrix, max):

    if(max):
        maxV = np.max(matrix)
        matrix *= -1
        matrix += maxV
    
    qx = quantityArrayX
    qy = quantityArrayY
    distribution = []
    filterMatrix = matrix.copy()

    # Repeat until all discounts are applied
    while not all([ v == 0 for v in qx ]):
        assignement = hungarianAlgorithm(filterMatrix)
        for val in assignement:
            distribution.append([val[1],  val[0],  min([qx[val[1]], qy[val[0]]])])
            qx[val[1]] -= qy[val[0]]
            if qx[val[1]] < 0:
                qy[val[0]] = qx[val[1]] * -1
                qx[val[1]] = 0
            else:
                qy[val[0]] = 0

        size = len(qx) - sum([ v == 0 for v in qx ])
        if size == 1:
            for x in range(len(qx)):
                for y in range(len(qx)):
                    if qx[x] != 0 and qy[y] != 0:
                        distribution.append([x, y, min([qx[x], qy[y]])])
                        qx[x] -= qy[y]
        else:
            filterMatrix = []
            for i in range(size):
                filterMatrix.append([0] * size)

            i = 0
            for x in range(len(qx)):
                for y in range(len(qx)):
                    if qx[x] != 0 and qy[y] != 0:
                        filterMatrix[i//size][i%size] = matrix[x][y]
                        i+=1

            filterMatrix = np.array(filterMatrix)
    return distribution

def generateMatrix(baseprice, discount, cvr):
    size = len(discount)
    dataM = []
    for i in range(size):
        dataM.append([0] * size)
    for i in range(len(discount)):
        for j in range(len(cvr)):
            dataM[i][j] = (baseprice-discount[i]) * cvr[j]
    
    return np.array(dataM)

def productDist(a, b, baseprice, discounts, cvr):
    for i in range(len(discounts)):
        discounts[i] = discounts[i] * -1
    return sortingByHungarian(a.copy(),b.copy(),generateMatrix(baseprice, discounts, cvr),True)


"""
# Discounts amounts
a = np.array([40,30,20,10])
# Customer Type amounts
b = np.array([40,30,20,10])


# Discounts amounts
a = np.array([40,30,20,10])
# Customer Type amounts
b = np.array([40,30,20,10])

# Items baseprice
baseprice = 20
# Discounts values
ds = [0,3,6,9]
# Conv. rates
cvs = [35,25,60,90]

# i - Promotion id, j - Customer type id
m = generateMatrix(baseprice, ds, cvs)
print(sortingByHungarian(a,b,m,True))
"""
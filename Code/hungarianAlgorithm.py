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
            if zeros[0:4].count(0) >= zeros[4:8].count(0):
                actual = zeros[0:4].index(max(zeros[0:4]))
            else:
                actual = zeros[4:8].index(max(zeros[4:8])) + 4

            if actual < m.shape[0]:
                crossX[actual] = True
            else:
                crossY[actual-m.shape[0]] = True
            
            zeros = [0 for x in range(m.shape[0]*2)]
            for i in range(m.shape[0]):
                for j in range(m.shape[0]):
                    if not crossX[i] and not crossY[j] and m[i][j] == 0:
                        zeros[i] += 1
                        zeros[4+j] += 1
                 
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

    print(m)
    while(crossX.count(False) + crossY.count(False) != 0):
        print (crossX, crossY)
        print("row",zeros[0:4],"col",zeros[4:8])
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
                if m[i,actual-4] == 0 and not crossX[i]:
                    crossX[i] = True
                    crossY[actual-4] = True
                    selections.append([i,actual-4])
                    break
        
        zeros = [0 for x in range(m.shape[0]*2)]
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                if not crossX[i] and not crossY[j] and m[i][j] == 0:
                    zeros[i] += 1
                    zeros[4+j] += 1
        print (selections)
    return m



def sortingByHungarian(quantityArrayX, quantityArrayY, matrix, max):

    if(max):
        maxV = np.max(matrix)
        matrix *= -1
        matrix += maxV

    qx = quantityArrayX
    qy = quantityArrayY
    print(matrix)

    assignement = hungarianAlgorithm(matrix)


a = np.array([40,30,20,10])
b = np.array([40,30,20,10])
baseprice = 20
ds = [0,3,6,9]
cvs = [35,25,60,90]

dataM = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
for i in range(len(ds)):
    for j in range(len(cvs)):
        dataM[i][j] = (20-ds[i]) * cvs[j]

m = np.array(dataM)
sortingByHungarian(a,b,m,True)
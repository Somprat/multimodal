import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    if len(a[0]) != len(b):
        return -1
    
    answer = [[0 for i in range(len(b[0]))] for j in range(len(a))]


    for i in range(len(a)):
        for j in range(len(b[0])):
            answer[i][j] = sum([a[i][k]*b[k][j] for k in range(len(b[0]))])

    return answer

print(matrixmul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]]))
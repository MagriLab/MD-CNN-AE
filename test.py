from numpy import array
for i in range(5):
    a = array([0,1,2,3,4,5])+i
    for j in range(6):
        if a[j] >= 4:
            print(j)
            break
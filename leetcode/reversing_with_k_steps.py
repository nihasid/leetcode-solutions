
def RightRotate( a, n, k):
    print("a: ", a, "N: ", n, "steps: ", k)
    for i in range(0, n):
 
        if(i < k):
 
            # Printing rightmost
            # kth elements
            print("if: ",a[n + i - k])
 
        else:
 
            # Prints array after
            # 'k' elements
            print("else: ",a[i - k])
 
    print("\n")
    

Array = [ 1, 2, 3, 4, 5 ]
N = len(Array)
K = 2
     
RightRotate(Array, N, K)


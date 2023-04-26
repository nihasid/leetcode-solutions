
def primeNum( n):
    # not done
    
    prime = ([x  for x in range(0, n) if n%x != 0 ])
    return prime
    
    

n = 20
print(primeNum(n))


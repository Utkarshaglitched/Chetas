import math
def dot(arr1,arr2):
    dot=0
    if len(arr1)==len(arr2):
        for i in range(len(arr1)):
            dot+=arr1[i]*arr2[i]

        return dot
    else:
        raise ValueError("Arrays must be of same size")
    
def mod(arr):
    all_sum=0
    for i in range(len(arr)):
        all_sum+=(arr[i]**2)
    
    return math.sqrt(all_sum)


def consimilaritry(a1,a2):
    costheta=dot(a1,a2)/(mod(a1)*mod(a2))
    return costheta

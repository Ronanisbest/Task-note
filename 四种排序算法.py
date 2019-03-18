##Insert sort

def insert_sort(ilist):
    for i in range(len(ilist)):
        for j in range(i):
            if ilist[i] < ilist[j]:
                ilist.insert(j, ilist.pop(i))
                break
    return ilist

##quick sort
def quick_sort(qlist):
    if qlist == []:
        return []
    else:
        qfirst = qlist[0]
        qless = quick_sort([l for l in qlist[1:] if l < qfirst])
        qmore = quick_sort([m for m in qlist[1:] if m >= qfirst])
        return qless + [qfirst] + qmore

##merge sort
def merge_sort(array):
    def merge_arr(arr_l, arr_r):
        array = []
        while len(arr_l) and len(arr_r):
            if arr_l[0] <= arr_r[0]:
                array.append(arr_l.pop(0))
            elif arr_l[0] > arr_r[0]:
                array.append(arr_r.pop(0))
        if len(arr_l) != 0:
            array += arr_l
        elif len(arr_r) != 0:
            array += arr_r
        return array
 
    def recursive(array):
        if len(array) == 1:
            return array
        mid = len(array) // 2
        arr_l = recursive(array[:mid])
        arr_r = recursive(array[mid:])
        return merge_arr(arr_l, arr_r)
 
    return recursive(array)
##
def sort(arr):
    result = []
    for index in range(0,len(arr)):
        result.append(0)
    for index in range(len(arr)):
        counter = result[arr[index]]+1
        result[arr[index]]=counter
    return result


arr = [1,3,5,7,9,2,9,4,6,8,0,1,1,3,2,2,2,2]
arr = sort(arr)
for item in range(len(arr)):
    if arr[item]!=0:
        step = arr[item]
        while step>0:
            print(item)
            step-=1
a = [1, 1, 1, 2, 2, 0, 0, 1, 2, 1, 0]

print(a)

for idx, i in enumerate(a):
    if i>1 : a[idx] = 1

print(a)
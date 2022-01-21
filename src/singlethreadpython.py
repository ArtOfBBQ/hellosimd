import random

vector_sizes = 50000000
print("preparing 2 lists of 50 million floats to work with...")

a = [0] * vector_sizes;
b = [0] * vector_sizes;
results = [0] * vector_sizes;

for i in range(vector_sizes):
    a[i] = random.randint(0, 150000) * 0.2
    b[i] = random.randint(0, 150000) * 0.2

print("calculating result..")
for i in range(vector_sizes):
    results[i] = (a[i] + b[i]) * b[i]


import random

# Create an array of numbers from 0 to 9 in a random distribution
numbers_array = [random.randint(0, 9) for _ in range(100)]

# Create a set to keep track of used indices
used_indices = set()
zero =[]
one = []
two=[]
three=[]
four=[]
five=[]
six=[]
seven=[]
eight=[]
nine=[]

# Loop through the range of indices from 0 to 9
for idx in range(10):
    # Find all indices of the current number in the numbers_array
    indices = [index for index, value in enumerate(numbers_array) if value == idx and index not in used_indices]
    if idx== 0:
        zero.append(indices)
    elif idx==1:
        one.append(indices)
    elif idx==2:
        two.append(indices)
    elif idx==3:
        three.append(indices)
    elif idx==4:
        four.append(indices)
    elif idx==5:
        five.append(indices)
    elif idx==6:
        six.append(indices)
    elif idx==7:
        seven.append(indices)
    elif idx==8:
        eight.append(indices)
    elif idx==9:
        nine.append(indices)



    # Add the indices to the used_indices set
    used_indices.update(indices)
    
    # Print the indices of the current number in ascending order
    for index in sorted(indices):
        print(f"Number: {idx}, Index: {index}")

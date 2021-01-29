# Write a python program to find the word counting a file for each line and then print the output.Finally store the output back to the file.

# Open the file which is the source folder with read access permission
infile = open("WordCountSecondFile", 'r')

# Create a dictionary variable which holds the word count output
count = {}
line = infile.readline()    # read first line

# Iterate through while loop until end of file is reached
while line != "":                       # If the line is not empty read through it
    for i in line.split(" "):           # Split each line according to the delimiter and check each word
        temp = (count.get(i))
        if temp is None:                # If the word is not present in the dictionary, add in it
            count[i] = 1
        else:                           # If the word is present in the dictionary, update it's count
            count.update({i: temp+1})
    line = infile.readline()            # Read next line

print("Word count = ") # Output the results iterating through the dictionary
for i in count:
    print(i, " ", count[i])

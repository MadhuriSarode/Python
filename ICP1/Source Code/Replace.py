# Write a program that accepts a sentence and replace each occurrence of ‘python’ with ‘pythons’ without using regex

string_a = input("Enter the string to be replaced: ")       # Accept the string from user
string_b = string_a.replace("python", "pythons")            # replace the python word with pythons
print("Original string = ", string_a)                       # print the string
print("Replaced string = ", string_b)                       # print replaced string

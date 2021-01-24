
# Get the input from user as a list of characters with spaces as separators
input_List_of_characters = input("Enter a list elements separated by space ")
print("\n")

# Split the user list
userList = input_List_of_characters.split()
print("user list is ", userList)

# Character list to string conversion
inputString = ""
for char in userList:
    inputString += char
print("input string = ", inputString)

# Deleting 2 characters
char_deleted_string = inputString[:inputString.index('y')] + inputString[inputString.index('y')+2:]
print("String with 2 characters deleted = ", char_deleted_string)

# Reversing the string
reversed_string = char_deleted_string[::-1]
print("Reversed string = ", reversed_string)


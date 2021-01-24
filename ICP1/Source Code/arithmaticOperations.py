# Take two numbers from user and perform arithmetic operations on them.
import math             # Math import for operators

num1 = int(input("Enter First Number: "))           # Input first number
num2 = int(input("Enter Second Number: "))          # Input second number

for x in range(6):                                  # Iterate the for loop for few times so user gets chance to perform many operations
    print("Enter which operation would you like to perform?")
    ch = input("Enter any of these char for specific operation +,-,*,/,sqrt,exponent,%,exit: ")

    result = 0
    result2 = 0
    if ch == '+':
        result = num1 + num2
    elif ch == '-':
        result = num1 - num2
    elif ch == '*':
        result = num1 * num2
    elif ch == '/':
        result = num1 / num2
    elif ch == 'sqrt':
        result = math.sqrt(num1)
        result2 = math.sqrt(num2)
    elif ch == '%':
        result = num1 % num2
    elif ch == 'exponent':
        result = num1 ** num2
    elif ch == 'exit':
        break
    else:
        print("Input character is not recognized!")

    if ch == 'sqrt':                            # Printing the result for square root function
        print(ch, num1, ":", result)
        print(ch, num2, ":", result2)
    else:
        print(num1, ch, num2, ":", result)      # Printing the result for other functions

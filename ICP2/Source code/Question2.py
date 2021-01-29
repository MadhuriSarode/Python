# Given a non-negative integer num, return the number of steps to reduce it to zero. If the current number is even, you have to divide it by 2, otherwise, you have to subtract 1 from it

# The output strings for even and odd numbers
even_str = " is even; divide by 2 and obtain"
odd_str = " is odd; subtract 1 and obtain"

# Input from the users for the number which has to be reduced to 0
input_num = int(input("Enter the number for the steps to reduce it to zero"))
num = input_num
step_count = 1

# Iterate through while loop till the number is not reduced to 0
while num != 0:
    if num % 2 == 0:                        # If number is even, divide by 2
        print("\nstep", step_count)
        print(num, even_str, num/2)
        num = num/2
    else:
        print("\nstep", step_count)         # If number is odd, subtract by 1
        print(num, odd_str, num-1)
        num = num-1
    step_count = step_count + 1             # Note the steps of each

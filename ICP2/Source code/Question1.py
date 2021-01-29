# Write a program, which reads height(feet.) of N students into a list and convert these heights to cm in a separate list:

# Input the number of students from user
students_count = int(input("Enter number of students "))
print("Enter the height of each student in feet ")

# Student list of heights in feet
student_height_feet = []

# Input each student's height which is added to the list
for i in range(students_count):
    print("Enter height : Student", (i+1))
    student_height_feet.append(float(input()))

print("The student's height in feet = ", student_height_feet)

# Student list of heights in cm
student_height_cm = []

# Iterate through each student's height and convert feet into cm
for j in range(students_count):
    student_height_cm.append(student_height_feet[j] * 30.48)            # Feet to cm conversion formula : 1 Feet = 30.48cm

print("Output = The student's height in cm = ", student_height_cm)

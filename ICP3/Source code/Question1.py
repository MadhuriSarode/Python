# Create a class Employee and then do the following


# Class showing color codes
class bColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# A function to calculate average salary, the total salary and number of employees are used.
def avg_salary():
    print("Total salary of all employees = ", Employee.total_salary)
    print("Total number of employees = ", Employee.no_of_employees)
    return Employee.total_salary / Employee.no_of_employees

# Employee class holing the data attributes and functions about employee details
class Employee:
    total_salary = 0     # data member for total salary
    no_of_employees = 0  # data member to count the number of Employees

    # constructor to initialize name, family, salary, department
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.no_of_employees += 1                       # Each time employee is created, it is counted
        Employee.total_salary += self.salary                # Salary of each employee is added up to total salary variable

    def get_employee_details(self):                         # Function to print employee details
        print("Name = ", self.name)
        print("Family =", self.family)
        print("Salary =", self.salary)
        print("Department =", self.department)


a = Employee("Madhuri", 3, 400, "IT")                       # Employee1
b = Employee("Karthik", 3, 400, "CS")                       # Employee2
print(bColors.HEADER + "Average salary = ", avg_salary())   # Average salary


# Full time employee class inheriting the data attributes and functions from Employee class
class FullTimeEmployee(Employee):
    def __init__(self, name, family, salary, department, state, ID):
        Employee.__init__(self, name, family, salary, department)
        self.state = state                                                  # Additional 2 data attributes state,ID
        self.ID = ID
        print("\nCreated Full time employee object")

    def get_fulltimeemployee_details(self):                                 # Function to print full time employee details
        print("Name = ", self.name)
        print("Family =", self.family)
        print("Salary = ", self.salary)
        print("Department =", self.department)
        print("State =", self.state)
        print("ID =", self.ID)


c = FullTimeEmployee("Aditya", 4, 900, "EC", "Dallas", 12345)               # Employee3
print(bColors.WARNING + "Results from accessing the inherited class full time employee function")
c.get_fulltimeemployee_details()

print(bColors.OKBLUE + "\nResults from accessing the parent class Employee function ")
c.get_employee_details()

# Import libraries
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
from bs4 import BeautifulSoup


class Employee:
    numberOfEmployees = 0  # Number of all Employees
    employees = []         # Employee array to recall later when needed

    def __init__(self, name, family, salary, department):
        self.name = name                # Name of the employee
        self.family = family            # Family of the employee
        self.salary = salary            # Salary of the employee
        self.department = department    # Department the employee works in

        # Add employees to the counter to recall later when needed
        Employee.numberOfEmployees += 1
        Employee.employees.append(self)

    # Get employee salary
    def get_salary(self):
        return self.salary

    # Get the average salary of all employees
    @staticmethod
    def average_salary():
        total_salaries = 0

        for employee in Employee.employees:
            total_salaries += int(employee.salary)

        return total_salaries / Employee.numberOfEmployees

    # Get employee name
    def get_name(self):
        return self.name


class FullTimeEmployee(Employee):
    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)


def parse_wiki_page():
    url = "https://en.wikipedia.org/wiki/Deep_learning"

    # Query the page
    opened_page = urllib2.urlopen(url)

    # Parse the page
    soup = BeautifulSoup(opened_page, 'html.parser')

    # Print title of the page
    print("Title:", soup.title.string, "\n")

    # Find all the a tags and print out the url
    for a in soup.find_all('a', href=True):
        print("link:", a['href'])


def main():
    # Gather information about the first employee
    print("Enter credentials for the first employee")
    employee_one_name = input("name: ")
    employee_one_family = input("family: ")
    employee_one_salary = input("salary: ")
    employee_one_department = input("department: ")

    # Gather information about the second employee
    print("Enter credentials for the second employee(Full time)")
    employee_two_name = input("name: ")
    employee_two_family = input("family: ")
    employee_two_salary = input("salary: ")
    employee_two_department = input("department: ")

    # Create instances of the employees
    employee_one = Employee(employee_one_name, employee_one_family, employee_one_salary, employee_one_department)
    employee_two = FullTimeEmployee(employee_two_name, employee_two_family, employee_two_salary, employee_two_department)

    # Print out salary information
    print("\nEmployee average salary")
    print(employee_one.get_name(), ":", employee_one.get_salary())
    print(employee_two.get_name(), ":", employee_two.get_salary())
    print("Average of", employee_one.get_name(), "and", employee_two.get_name(), "=", Employee.average_salary())

    # Parse the wiki page
    print("\n\n-----------------\n\nPART TWO\n")
    parse_wiki_page()


main()

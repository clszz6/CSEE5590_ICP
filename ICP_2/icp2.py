"""
This program will:
1. Take a number input from a user and calculate the average of the amount of numbers the user has decided.
2. Implement a stack/queue from a list.
3. Take a string from a user and reverse the cases of each character in the string
"""

from collections import deque
from decimal import Decimal


def average_number_user_inputs():
    # Get amount of numbers a user has and initialize the sum var
    amountOfNumbers = eval(input("How many numbers do you have? "))
    sumOfNumbers = 0.0

    # Loop through and sum up the amount of numbers the user specified
    listOfNumbers = list(map(int, input("Enter the numbers >> ").split()))
    for i in range(len(listOfNumbers)):
        sumOfNumbers = sumOfNumbers + listOfNumbers[i]

    # Get the average and print out to 3 decimal places
    average = Decimal(sumOfNumbers / amountOfNumbers)
    print("\nThe average of the numbers is:", round(average, 3))


def create_stack():
    # Get numbers from user to add to the stack
    stackOfNumbers = list(map(int, input("\nEnter the numbers you want in the stack >> ").split()))

    # Print out the top of the stack
    print("stack top: ", stackOfNumbers[len(stackOfNumbers) - 1])

    # Pop the stack and then print the results
    stackOfNumbers.pop()
    print("after stack pop: ", stackOfNumbers)

    # Push a number the user provides onto the stack and then print the results
    numberToPush = input("number to push onto the stack: ")
    stackOfNumbers.append(int(numberToPush))
    print("stack now equals: ", stackOfNumbers)


def create_queue():
    # Get names from user to add to the queue
    queueOfNames = deque(map(str, input("\nEnter the names you want in the queue >> ").split()))

    # Enqueue a name onto the queue that the user provides
    nameToEnque = input("name to enqueue onto the queue: ")
    queueOfNames.append(nameToEnque)
    print("queue after enqueue: ", queueOfNames)

    # dequeue the queue by one and then print the results
    queueOfNames.popleft()
    print("queue after dequeue: ", queueOfNames)


def reverse_string_user_provides():
    # Get a string from a user
    stringToReverseCases = input("\nEnter a string: ")

    # print out the reversed cases in the string with the built in swapcase() method
    print(stringToReverseCases.swapcase())


def main():
    average_number_user_inputs()
    create_stack()
    create_queue()
    reverse_string_user_provides()


main()

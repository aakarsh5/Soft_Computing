import pandas as pd
import random

# Constants
NUM_STUDENTS = 1000
SUBJECTS = ['Math', 'Science', 'English', 'History', 'Computer']
PASS_MARK = 50

# Sample names
first_names = [
    'Aman', 'Neha', 'Ravi', 'Priya', 'Karan', 'Simran', 'Rahul', 'Anjali', 'Vikram', 'Sneha',
    'Arjun', 'Ishita', 'Kabir', 'Divya', 'Siddharth', 'Meera', 'Tanish', 'Pooja', 'Yash', 'Tanvi',
    'Lakshmi', 'Harsh', 'Sanya', 'Abhay', 'Naina', 'Rohit', 'Sakshi', 'Ishan', 'Aarav', 'Kiara'
]

last_names = [
    'Sharma', 'Verma', 'Kumar', 'Yadav', 'Singh', 'Patel', 'Gupta', 'Mishra', 'Reddy', 'Joshi',
    'Nair', 'Chopra', 'Mehta', 'Pillai', 'Desai', 'Malhotra', 'Bansal', 'Kapoor', 'Trivedi', 'Chatterjee'
]
def generate_random_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

def generate_pass_marks():
    # Generate marks where all subjects are >= 50
    return [random.randint(50, 100) for _ in SUBJECTS]

def generate_fail_marks():
    # Make at least one subject < 50
    marks = [random.randint(50, 100) for _ in SUBJECTS]
    fail_index = random.randint(0, len(SUBJECTS)-1)
    marks[fail_index] = random.randint(0, 30)
    return marks

data = []

for _ in range(NUM_STUDENTS):
    name = generate_random_name()
    if random.random() < 0.7:  
        marks = generate_pass_marks()
        result = 1
    else:  
        marks = generate_fail_marks()
        result = 0
    record = [name] + marks + [result]
    data.append(record)

columns = ['Name'] + SUBJECTS + ['Final_Result']
df = pd.DataFrame(data, columns=columns)

df.to_csv('student_results.csv', index=False)

print("CSV file 'student_results.csv' created with 70% passing students.")

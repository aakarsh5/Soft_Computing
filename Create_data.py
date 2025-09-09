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

# Generate random full name
def generate_random_name():
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Generate passing marks
def generate_pass_marks():
    return [random.randint(50, 100) for _ in SUBJECTS]

# Generate failing marks
def generate_fail_marks():
    marks = [random.randint(50, 100) for _ in SUBJECTS]
    fail_index = random.randint(0, len(SUBJECTS)-1)
    marks[fail_index] = random.randint(0, 30)  # force fail
    return marks

# Function to calculate grade
def calculate_grade(avg):
    if avg >= 90:
        return 'A+'
    elif avg >= 80:
        return 'A'
    elif avg >= 70:
        return 'B'
    elif avg >= 60:
        return 'C'
    elif avg >= 50:
        return 'D'
    else:
        return 'F'

data = []

for _ in range(NUM_STUDENTS):
    name = generate_random_name()
    
    # 70% pass, 30% fail
    if random.random() < 0.7:
        marks = generate_pass_marks()
        result = 1
    else:
        marks = generate_fail_marks()
        result = 0
    
    avg_marks = sum(marks) / len(SUBJECTS)
    grade = calculate_grade(avg_marks)
    
    record = [name] + marks + [result, avg_marks, grade]
    data.append(record)

columns = ['Name'] + SUBJECTS + ['Final_Result', 'Average_Marks', 'Course_Grade']
df = pd.DataFrame(data, columns=columns)

df.to_csv('student_results_with_grades.csv', index=False)

print("CSV file 'student_results_with_grades.csv' created with course grades.")

import sqlite3
import random

conn = sqlite3.connect("school.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
cursor.execute("CREATE TABLE IF NOT EXISTS courses (id INTEGER PRIMARY KEY, name TEXT, credits INTEGER)")
cursor.execute(
    "CREATE TABLE IF NOT EXISTS grades (student_id INTEGER, course_id INTEGER, grade REAL, FOREIGN KEY (student_id) REFERENCES students (id), FOREIGN KEY (course_id) REFERENCES courses (id))")
cursor.execute("CREATE TABLE IF NOT EXISTS teachers (id INTEGER PRIMARY KEY, name TEXT, salary REAL)")
cursor.execute(
    "CREATE TABLE IF NOT EXISTS assignments (id INTEGER PRIMARY KEY, name TEXT, course_id INTEGER, deadline TEXT, FOREIGN KEY (course_id) REFERENCES courses (id))")

cursor.execute("ALTER TABLE courses ADD COLUMN teacher_id INTEGER REFERENCES teachers (id)")

class Student:
    def __init__(self, id, name, age):
        self.id = id
        self.name = name
        self.age = age

    def enroll(self, course):
        grade = round(random.uniform(1.0, 5.0), 2)
        cursor.execute("INSERT INTO grades VALUES (?, ?, ?)", (self.id, course.id, grade))

    def __str__(self):
        return f"Студент {self.id}: {self.name}, {self.age} лет"


class Course:
    def __init__(self, id, name, credits):
        self.id = id
        self.name = name
        self.credits = credits

    def assign(self, teacher):
        cursor.execute("UPDATE courses SET teacher_id = ? WHERE id = ?", (teacher.id, self.id))

    def create_assignment(self, name, deadline):
        cursor.execute("INSERT INTO assignments (name, course_id, deadline) VALUES (?, ?, ?)",
                       (name, self.id, deadline))

    def __str__(self):
        return f"Курс {self.id}: {self.name}, {self.credits} кредит"


class Teacher:
    def __init__(self, id, name, salary):
        self.id = id
        self.name = name
        self.salary = salary

    def __str__(self):
        return f"Учитель {self.id}: {self.name}, {self.salary} рублей в месяц"


class Assignment:
    def __init__(self, id, name, course_id, deadline):
        self.id = id
        self.name = name
        self.course_id = course_id
        self.deadline = deadline

    def __str__(self):
        return f"Назначение {self.id}: {self.name}, для курса {self.course_id}, до {self.deadline}"



def generate_name():
    first_name = random.choice(
        ["Алексей", "Анна", "Борис", "Вера", "Григорий", "Дарья", "Евгений", "Елена", "Иван", "Ирина", "Кирилл",
         "Людмила", "Михаил", "Надежда", "Олег", "Ольга", "Павел", "Раиса", "Сергей", "Татьяна"])
    last_name = random.choice(
        ["Иванов", "Петров", "Сидоров", "Смирнов", "Попов", "Васильев", "Соколов", "Новиков", "Федоров", "Морозов",
         "Волков", "Алексеев", "Лебедев", "Семенов", "Егоров", "Павлов", "Козлов", "Степанов", "Орлов", "Андреев"])
    return first_name + " " + last_name


def generate_date():
    year = random.randint(2023, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"



courses = [
    Course(1, "Математика", 4),
    Course(2, "Физика", 3),
    Course(3, "Информатика", 5),
    Course(4, "Литература", 2),
    Course(5, "История", 3)
]


teachers = [
    Teacher(1, "Виктор Смирнов", 50000),
    Teacher(2, "Мария Петрова", 45000),
    Teacher(3, "Андрей Иванов", 60000),
    Teacher(4, "Елена Соколова", 40000),
    Teacher(5, "Дмитрий Новиков", 55000)
]

students = []
for i in range(1, 21):
    name = generate_name()
    age = random.randint(18, 25)
    student = Student(i, name, age)
    students.append(student)


for course in courses:
    cursor.execute("INSERT INTO courses (id, name, credits) VALUES (?, ?, ?)", (course.id, course.name, course.credits))

    teacher = random.choice(teachers)
    course.assign(teacher)

    for j in range(1, 3):
        name = f"Задание {j} по {course.name}"
        deadline = generate_date()
        course.create_assignment(name, deadline)

for teacher in teachers:
    cursor.execute("INSERT INTO teachers (id, name, salary) VALUES (?, ?, ?)",
                   (teacher.id, teacher.name, teacher.salary))

for student in students:
    cursor.execute("INSERT INTO students (id, name, age) VALUES (?, ?, ?)", (student.id, student.name, student.age))

    for k in range(3):
        course = random.choice(courses)
        student.enroll(course)


conn.commit()


print("Запрос 1: Вывести имена и оценки студентов по курсу Физика")
cursor.execute(
    "SELECT students.name, grades.grade FROM students JOIN grades ON students.id = grades.student_id JOIN courses ON grades.course_id = courses.id WHERE courses.name = 'Физика'")
results = cursor.fetchall()
for row in results:
    print(row[0], row[1])
print()

print("Запрос 2: Вывести имена преподавателей и названия курсов, которые они ведут")
cursor.execute("SELECT teachers.name, courses.name FROM teachers JOIN courses ON teachers.id = courses.teacher_id")
results = cursor.fetchall()
for row in results:
    print(row[0], row[1])
print()

conn.close()
import time

class ThreeStates:
    # Определяем конструктор класса
    def __init__(self):
        self.__state = 1

    def hasstate(self):
        while False:
            if self.__state == 1:
                print("Состояние 1")
                time.sleep(5)
                self.__state = 2
            elif self.__state == 2:
                print("Состояние 2")
                time.sleep(3)
                self.__state = 3
            elif self.__state == 3:
                print("Состояние 3")
                time.sleep(2)
                self.__state = 1

ts = ThreeStates()
ts.hasstate()

class Volume:

    def __init__(self, length, width, height):
        self._length = length
        self._width = width
        self._height = height

    def volume(self):
        return self._length * self._width * self._height


rectangle = Volume(10, 20, 30)
print(rectangle.volume())

class Employee:

    def __init__(self, name, patronymic, surname, salary):
        self.name = name
        self.patronymic = patronymic
        self.surname = surname
        self.salary = salary


class Salary(Employee):

    def __init__(self, name, patronymic, surname, salary):
        super().__init__(name, patronymic, surname, salary)

    def get_full_name(self):
        return f'\n{self.surname} {self.name} {self.patronymic}'

    def get_total_income(self):
        return self.salary["wage"] + self.salary["bonus"]


s1 = Salary("Сидоров", "Игорь", "Павлович", {"wage": 25000, "bonus": 10000})
s2 = Salary("Иванов", "Иван", "Олегович", {"wage": 45000, "bonus": 15000})

print(s1.get_full_name())
print(s1.get_total_income())

print(s2.get_full_name())
print(s2.get_total_income())


class Airplane:

    def __init__(self, speed, color, name, is_jet):
        self.speed = speed
        self.color = color
        self.name = name
        self.is_jet = is_jet

    def go(self):
        print(f"{self.name} летит ")

    def stop(self):
        print(f"{self.name} не летит")

    def direction(self, angle):
        print(f"{self.name} повернул на {angle} градусов")

    def show_speed(self):
        print(f"Текущая скорость {self.name} - {self.speed} км/ч")


class FastAirplane(Airplane):
    def __init__(self, speed, color, name, is_jet):
        super().__init__(speed, color, name, is_jet)

    def show_speed(self):
        if self.speed > 1300:
            print(f"{self.name} летит со сверхзвуковой скоростью {self.speed} км/ч")
        else:
            super().show_speed()


class Biplane(Airplane):
    def __init__(self, speed, color, name, is_jet):
        super().__init__(speed, color, name, is_jet)


class ArmyAirplane(Airplane):
    def __init__(self, speed, color, name, is_jet):
        super().__init__(speed, color, name, is_jet)

plane1 = Airplane(1000, "черный", "какое-то название", True)
plane2 = FastAirplane(2000, "белый", "боинг", False)
plane3 = Biplane(700, "бежевый", "МС-21", True)
plane4 = ArmyAirplane(900, "зеленый", "АН-2", False)

print()
plane1.go()
plane2.show_speed()
plane3.direction(35)
plane4.stop()


class MathOperations:
    def __init__(self, first_num, second_num):
        self.first_num = first_num
        self.second_num = second_num

    def calc(self):
        print("\nЗапуск операции")


class my_sum(MathOperations):
    def calc(self):
        super().calc()
        return self.first_num + self.second_num


class my_sub(MathOperations):
    def calc(self):
        super().calc()
        return self.first_num - self.second_num


class my_mult(MathOperations):
    def calc(self):
        super().calc()
        return self.first_num * self.second_num

a = my_sum(5, 7)
b = my_sub(5, 7)
c = my_mult(5, 7)

print(a.calc())
print(b.calc())
print(c.calc())

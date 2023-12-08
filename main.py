import math


def solve_equation(equation):
    try:
        value = eval(equation)
        value = round(value, 3)
        return value
    except ZeroDivisionError:
        return "На ноль делить нельзя!"
    except ValueError:
        return "Функция логарифма не определена в нуле или для отрицательных чисел!"


x = float(input("Введите значение переменной x:"))
y = float(input("Введите значение переменной y:"))
z = float(input("Введите значение переменной z:"))

# Первое уравнение
k0 = solve_equation("math.log((y - math.sqrt(abs(x))) * (x - y / (z + (x**2) / 4)))")
k = solve_equation("math.log(abs(k0))")
print(f"1.Значение переменной k:\n {k}")

# Второе уравнение
d = solve_equation("2 * math.cos(x - math.pi / 6) /(0.5 + math.pow(math.sin(y)+abs(y - x)/3,2))")
print(f"2.Значение переменной d:\n {d}")

# Третье уравнение
w = solve_equation(
    "((x / y) * (z + x) * math.exp(abs(x - y)) + math.log(1 + math.e)) /(math.pow(math.sin(y),2) - math.pow(math.sin(x) * math.sin(y),2))")
print(f"3.Значение переменной w:\n {w}")

# Четвёртое уравнение
b = solve_equation("(3 + math.exp(y - 1)) / (1 + (x**2) * abs(y - math.tan(z)))")
print(f"4.Значение переменной b:\n {b}")

# Пятое уравнение
p = solve_equation(
    "math.sqrt(abs(x * y)) + 2 * z if x * y < -2 else (x**3 + y**2 - z**2) if (x * y >= -2) and (x * y <= 2) else (math.pow(x,z) - y)")
print(f"5.Значение переменной p:\n {p}")

# Шестое уравнение
h = solve_equation("math.atan(x + abs(y)) if x < y else (math.atan(abs(x) + y) if x > y else (math.pow(x + y,2)))")
print(f"6.Значение переменной h:\n {h}")

# Седьмое уравнение
b1 = solve_equation(
    "math.log(x / y) + math.pow(x**2 + y, 3) if y != 0 and x / y > 0 else (math.log(abs(x / y)) + math.pow(x**2 + y, 3) if y != 0 and x / y < 0 else (math.pow(x**2 + y, 3) if x == 0 else 0))")
print(f"7.Значение переменной b1:\n {b1}")

# Восьмое уравнение
b2 = solve_equation(
    "math.sin(x + y) + 2 * math.pow(x + y, 2) if (x - y) > 0 else (abs(x**2 + math.sqrt(y)) if (y != 0) and (x == 0) else (0 if y == 0 else (math.sin(x - y) + math.pow(x - y, 3))))")
print(f"8.Значение переменной b2:\n {b2}")

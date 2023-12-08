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

# Массив с выражениями
expressions = {
    'k': "math.log(abs(math.log((y - math.sqrt(abs(x))) * (x - y / (z + (x**2) / 4)))))",
    'd': "2 * math.cos(x - math.pi / 6) /(0.5 + math.pow(math.sin(y)+abs(y - x)/3,2))",
    'w': "((x / y) * (z + x) * math.exp(abs(x - y)) + math.log(1 + math.e)) /(math.pow(math.sin(y),2) - math.pow(math.sin(x) * math.sin(y),2))",
    'b': "(3 + math.exp(y - 1)) / (1 + (x**2) * abs(y - math.tan(z)))", # b
    'p': "math.sqrt(abs(x * y)) + 2 * z if x * y < -2 else (x**3 + y**2 - z**2) if (x * y >= -2) and (x * y <= 2) else (math.pow(x,z) - y)",
    'h': "math.atan(x + abs(y)) if x < y else (math.atan(abs(x) + y) if x > y else (math.pow(x + y,2)))",
    'b': "math.log(x / y) + math.pow(x**2 + y, 3) if y != 0 and x / y > 0 else (math.log(abs(x / y)) + math.pow(x**2 + y, 3) if y != 0 and x / y < 0 else (math.pow(x**2 + y, 3) if x == 0 else 0))",
    'b': "math.sin(x + y) + 2 * math.pow(x + y, 2) if (x - y) > 0 else (abs(x**2 + math.sqrt(y)) if (y != 0) and (x == 0) else (0 if y == 0 else (math.sin(x - y) + math.pow(x - y, 3))))"
}

for i in expressions:
    result = solve_equation(expressions[i])
    print(f"Значение переменной {i}: {result}")

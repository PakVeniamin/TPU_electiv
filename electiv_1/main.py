import functools

# Вводим числа
a = input()
b = input()
c = input()

# Перевод из строкового типа данных в числовой
list_of_numbers = [float(x) if '.' in x else int(x) for x in (a, b, c)]


def diff_numbers(list):
    return functools.reduce(lambda x, y: x-y, list)


def multiplication(list):
    return functools.reduce(lambda x, y: x*y, list)


def quot(list):
    return functools.reduce(lambda x, y: x/y, list)


summ = sum(list_of_numbers)
diff = diff_numbers(list_of_numbers)
multip = multiplication(list_of_numbers)
quot = quot(list_of_numbers)

# Си-стиль
print("Си-стиль")
print("Сумма: %.3f" % summ)
print("Разность: %.3f" % diff)
print("Произведение: %.3f" % multip)
print("Частное: %.3f\n" % quot)

# Метод format
print("Метод format")
print("Сумма: {:.3f}".format(summ))
print("Разность: {:.3f}".format(diff))
print("Произведение: {:.3f}".format(multip))
print("Частное: {:.3f}\n".format(quot))

# f-строки
print("f-строки")
print(f"Сумма: {summ:.3f}")
print(f"Разность: {diff:.3f}")
print(f"Произведение: {multip:.3f}")
print(f"Частное: {quot:.3f}")
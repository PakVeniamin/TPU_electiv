import random
import time

'''Напишу декоратор чтобы считать время выполнения фунции'''


def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        func_result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return f'{args[1]}\nРезультат сортировки: {func_result} Время: {end_time - start_time}\n'

    return wrapper


numbers = [random.randint(-20, 20) for i in range(20)]


@time_decorator
def selection_sort(numbers: list[int], name: str) -> list[int]:
    for i in range(len(numbers) - 1):
        min_index = i
        for j in range(i + 1, len(numbers)):
            if numbers[j] < numbers[min_index]:
                min_index = j

        if min_index != i:
            numbers[min_index], numbers[i] = numbers[i], numbers[min_index]
    return numbers


print(selection_sort(numbers, 'Сортировка выбором'))


@time_decorator
def insertion_sort(numbers: list[int], name) -> list[int]:
    for i in range(1, len(numbers)):
        current_value = numbers[i]
        j = i - 1
        while j >= 0:
            if numbers[j] > current_value:
                numbers[j + 1] = numbers[j]
                numbers[j] = current_value
                j -= 1
            else:
                break

    return numbers


print(insertion_sort(numbers, 'Сортировка вставками'))


@time_decorator
def bubble_sort(numbers: list[int], name) -> list[int]:
    for i in range(len(numbers)):
        for j in range(len(numbers) - i - 1):
            if numbers[j] > numbers[j + 1]:
                numbers[j], numbers[j + 1] = numbers[j + 1], numbers[j]
    return numbers


print(bubble_sort(numbers, 'Пузырьковая сортировка'))


@time_decorator
def shell_sort(numbers: list[int], name) -> list[int]:
    gap = len(numbers) // 2

    while gap > 0:
        for i in range(gap, len(numbers)):
            current_value = numbers[i]
            position = i

            while position >= gap and numbers[position - gap] > current_value:
                numbers[position], numbers[position - gap] = numbers[position - gap], numbers[position]
                position -= gap

        gap //= 2
    return numbers


print(shell_sort(numbers, 'Сортировка Шелла'))


def quick_sort(numbers: list[int]) -> list[int]:
    n = len(numbers)
    if n <= 1:
        return numbers
    else:
        pivot = numbers[len(numbers)//2]
        less = [x for x in numbers if x < pivot]
        greater_or_equal = [x for x in numbers if x > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater_or_equal)

start_time = time.perf_counter()
result_quick_sort = quick_sort(numbers)
end_time = time.perf_counter()

print(f'Результат сортировки: {result_quick_sort} Время: {end_time - start_time}')


'''4 Самостоятельная'''


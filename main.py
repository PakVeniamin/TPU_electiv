import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
#
random_numbers = np.random.uniform(low=-9.9, high=9.9, size=196)
array = np.array(random_numbers).reshape(14, 14)
array = np.round(array, decimals=1)

'''Бинаризация'''
bin_data = preprocessing.Binarizer(threshold=2.1).transform(array)
print(f'Binarized data:\n {bin_data}\n')

'''Исключение среднего'''
print('Исключение среднего')
print(array.mean(axis=0))
print(array.std(axis=0))

'значение в массиве минус среднее и делим на отклонение'
data_scale = preprocessing.scale(array)
print('After\n')
print(data_scale.mean(axis=0))
print('Std deviation= ', data_scale.std(axis=0))

'''Масштабирование'''
'''делает атк что значения в массиве становятся от 0 до 1
это нужно чтобы разные признаки имели одинаковый масштаб
берем самое маленькое и самое большое число, делаем следующее:
Берем каждое число и вычитаем самое маленькое, потом делим
полученную на разность самого большого и маленького
в итоге самое маленькое станет 0, а самое большое станет 1'''
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(array)
print(f'Масштабирование\n{data_scaled_minmax}\n')

'''Нормализация'''
'''l1 - метод наименьших абсолютных отклонений
l2 - метод наименьших квадратов
Норма L1 - это сумма абсолютных значений всех чисел в строке.
Норма L2 - это квадратный корень из суммы квадратов всех чисел в строке.
помогает сделать так,
что разные признаки имею одинаковую важность и для их измерения можно использовать одну шкалу'''

print(f' before \n {np.sum(np.abs(array), axis=1)}\n')
data_normalized_11 = preprocessing.normalize(array, norm='l1')
data_normalized_12 = preprocessing.normalize(array, norm='l2')

print(f' after \n {np.sum(np.abs(data_normalized_11), axis=1)}')
print(f' after \n {np.sum(np.square(data_normalized_12), axis=1)}')
print(f'L1 normalized\n {data_normalized_11}')
print(f'L2 normalized\n {data_normalized_12}')


'''Кодирование'''
'''encoder это объект который позволяет нам переводить слова в числа и обратно
с помощью classes_ мы можем посмотреть какие слова он запомнил
.classes ведет список всех уникальных слов отсортированный по возрастанию
transform заменяет каждое слово в списке на число'''

'''кодирование нужно для того, чтобы преобразовать какие-то элементы в числа
это нужно для того, чтобы тренировать модель'''
print('Кодирование')
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

encoded_values = encoder.transform(input_labels)
print('\nLabels =', input_labels)
print('Encoded values', list(encoded_values))

encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print('\nEncoded values =', encoded_values)
print('Decoded labels =', decoded_list)


'''Логические классификатор'''

import matplotlib.pyplot as plt
from utilites import visualize_classifier
'''логический классификатор это функция которая умеет разделять объекты на разные классы
по их признакам в data каждая строка это объект, а столбец - признак.
у - набор классов, у нас их 4 штуки

solver - опредлеяет метод обучения, C = определяет насколько сильно классификатор будет
пытаться избегать ошибки, например C=1 сделает так что классификатор будет искать
баланс между точностью и обобщением'''
data = np.array(
    [[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4],
     [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
classifier1 = linear_model.LogisticRegression(solver='liblinear', C=1)
classifier1.fit(data, y)
visualize_classifier(classifier1, data, y)

'''Наивный Байес'''
'''Наивный предполагает что все признаки независимы друг от друга'''
print('Наивный Байес')
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

classfier = GaussianNB()
'обучаем классификатор и предиктим классы для объектов'
classfier.fit(X, y)
y_pred = classfier.predict(X)
'вычисляем долю правильных предиктов'
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print('Точность классификатора наивного байеса', round(accuracy, 2), "%")
visualize_classifier(classfier, X, y)
'''считаем точность, полноту и F меру для этого используем cross_val_score
которая оценивает классификатор на разных частях данных
cv = 3 означает что мы делим данные на 3 части и по очереди используем одну из них для тестирования
а две другие для обучения'''
print("precision: ", round(100 * cross_val_score(classfier, X, y, cv=3, scoring='precision_weighted').mean(), 2))
print("recall: ", round(100 * cross_val_score(classfier, X, y, cv=3, scoring='recall_weighted').mean(), 2))
print("f1: ", round(100 * cross_val_score(classfier, X, y, cv=3, scoring='f1_weighted').mean(), 2))

'''Матрица неточностей'''

print('Матрица неточностей')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

matrix = confusion_matrix(y, y_pred)
print(matrix)

print(classification_report(y, y_pred))

'''Машины опорных векторов'''
'классификатор который определяется с использованием гиперплоскости, которая разделяет классы'
'''Машины опорных векторов (SVM) - это тип алгоритма контролируемого обучения, который может быть использован для задач классификации и регрессии. Они особенно полезны для решения задач с высокоразмерными данными, когда количество признаков намного больше, чем количество образцов. Основная идея метода заключается в построении гиперплоскости, разделяющей объекты выборки оптимальным способом1.
Гиперплоскость - это пространство размерности n-1, которое может разделить пространство размерности n на две части. Например, в двумерном пространстве гиперплоскость - это прямая, а в трехмерном - плоскость. Гиперплоскость может быть линейной или нелинейной, в зависимости от того, как она описывается. Линейная гиперплоскость имеет вид:
w^T * x+b=0
где w - это вектор нормали к гиперплоскости, x - это вектор признаков объекта, а b - это смещение гиперплоскости от начала координат. 
Машины опорных векторов стремятся найти такую гиперплоскость, которая максимизирует зазор между классами, то есть расстояние от гиперплоскости до ближайших к ней объектов разных классов. Эти объекты называются опорными векторами, так как они определяют положение и ориентацию гиперплоскости. Чем больше зазор, тем лучше обобщающая способность модели, то есть ее способность правильно классифицировать новые данные3.'''

print('Машины опорных векторов')
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

input = 'income_data.txt'

with open(input) as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

X = np.array(X)
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
'''Этот код создает объект классификатора, который использует стратегию один против всех (OvR) 
для многоклассовой классификации. Это означает, что для каждого класса обучается 
один бинарный классификатор, который отделяет этот класс от всех остальных. 
Для предсказания класса нового примера выбирается классификатор, который дает 
наибольшее значение функции решения'''
'''OneVsRestClassifier - это класс из библиотеки scikit-learn, который позволяет использовать любой бинарный классификатор для реализации стратегии OvR. В этом случае в качестве базового классификатора используется LinearSVC2.

LinearSVC - это еще один класс из библиотеки scikit-learn, который реализует линейную 
поддержку векторной классификации (Linear Support Vector Classification). Это метод, 
который находит оптимальную гиперплоскость, разделяющую два класса в пространстве признаков.
Гиперплоскость выбирается таким образом, чтобы максимизировать расстояние до ближайших точек
 каждого класса, называемых опорными векторами'''
classifier3 = OneVsRestClassifier(LinearSVC(random_state=0))
classifier3.fit(X, y)

'''train_test_split - это функция из библиотеки scikit-learn, которая позволяет делать такое разделение с различными параметрами12. В этом коде заданы следующие параметры:
test_size=0.2: это означает, что 20% данных будут отведены для тестового подмножества, а остальные 80% - для обучающего.
random_state=5: это означает, что данные будут перемешаны случайным образом перед 
разделением, но с фиксированным значением генератора случайных чисел. Это нужно для того, 
чтобы результат был воспроизводимым при повторном запуске кода.'''
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=5)

classifier4 = OneVsRestClassifier(LinearSVC(random_state=0))
classifier4.fit(X_train, y_train)
y_test_pred = classifier4.predict(X_test)

f1 = cross_val_score(classifier4, X, y, cv=3, scoring='f1_weighted')
print(round(100 * f1.mean(), 2))

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family',
              'White', 'Male', '0', '0', '40', 'United-States']
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count += 1

input_data_encoded = np.array([input_data_encoded])
predicted_classes = classifier4.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_classes)[0])

'''Создание регрессора одной переменной'''
print('Создание регрессора одной переменной')
'''регрессия это процесс оценки того, как соотносятся входящие и выходящие переменные'''

input_data = 'data_singlevar_regr.txt'

data = np.loadtxt(input_data, delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.show()


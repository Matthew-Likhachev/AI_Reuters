#46 themes about news line
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)
#аргумент num_words=10000 ограничивает данные 
#10 000 наиболее часто встречающимися словами


#print(len(train_data))
#8982
#print(len(test_data))
#2246
#=>Всего у нас имеется 8982 обучающих и 2246 контрольных примеров


#print(train_data[10])
#каждый пример — это список целых чисел (индексов слов)
#[1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979, 3554, 14, 46, 4689, 4329, 86, 61, 3499, 4795, 14, 61, 451, 4329, 17, 12]


#Декодирование новостей обратно в текст если надо
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.
items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
 train_data[0]])
#Обратите внимание, что индексы смещены на 3, потому что индексы 
#, 1 и 2 зарезервированы для слов «padding» (отступ), «start of 
#sequence» (начало последовательности) и «unknown» (неизвестно)

#print(decoded_newswire)
#said as a result of its december acquisition of space co it expects earnings per share in 1987 of 1 15 to 1 30 dlrs per share up from 70 cts in 1986 the company said pretax net should rise to nine to 10 mln dlrs from six mln dlrs in 1986 and rental operation revenues to 19 to 22 mln dlrs from 12 5 mln dlrs it said cash flow per share this year should be 2 50 to three dlrs reuter 3

#Кодирование данных
#Векторизиврование данных
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension)) #vector with all zeros
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.   #если какое-то знач то ставим единицу
    return results
x_train = vectorize_sequences(train_data) #Векторизованные обучающие данные
x_test = vectorize_sequences(test_data)   #Векторизованные контрольные данны

#Векторизирование меток
#Векторизовать метки можно одним из двух способов: сохранить их в тензоре 
#целых чисел или использовать прямое кодирование. Прямое кодирование (onehot encoding) широко используется для форматирования категорий и также называется кодированием категорий (categorical encoding). Более подробно прямое 
#кодирование объясняется в разделе 6.1. В данном случае прямое кодирование меток 
#заключается в конструировании вектора с нулевыми элементами со значением 1 
#в элементе, индекс которого соответствует индексу метки. Например:
#def to_one_hot(labels, dimension=46):
#    results = np.zeros((len(labels), dimension))
#    for i, label in enumerate(labels):
#         results[i, label] = 1.
#    return results

#one_hot_train_labels = to_one_hot(train_labels)
#one_hot_test_labels = to_one_hot(test_labels)
#Следует отметить, что этот способ уже реализован в Keras, как мы видели в примере MNIST:
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels) 
one_hot_test_labels = to_categorical(test_labels) 


#Определение модели
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
#1. Сеть завершается слоем Dense с размером 46. Это означает, что для каждого 
#входного образца сеть будет выводить 46-мерный вектор. Каждый элемент этого 
#вектора (каждое измерение) представляет собой отдельный выходной класс.

#2. Последний слой использует функцию активации softmax. Мы уже видели этот 
#шаблон в примере MNIST. Он означает, что сеть будет выводить распределение 
#вероятностей по 46 разным классам — для каждого образца на входе сеть будет 
#возвращать 46-мерный вектор, где output[i] — вероятность принадлежности 
#образца классу i. Сумма 46 элементов всегда будет равна 1.

#Лучшим вариантом в данном случае является использование функции потерь 
#categorical_crossentropy. Она определяет расстояние между распределениями 
#вероятностей: в данном случае между распределением вероятности на выходе 
#сети и истинным распределением меток. Минимизируя расстояние между этими 
#двумя распределениями, мы учим сеть выводить результат, максимально близкий 
#к истинным меткам



#Компиляция модели
model.compile(optimizer='rmsprop',
 loss='categorical_crossentropy',
 metrics=['accuracy'])

#. Создание проверочного набора
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]



#Обучение модели
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=6,
    batch_size=512,
    validation_data=(x_val, y_val))



#Формирование графиков потерь на этапах обучения и проверки
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Формирование графиков точности на этапах обучения и проверки
plt.clf() #Очистить рисунок
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, one_hot_test_labels)
print(results)

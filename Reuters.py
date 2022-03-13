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

#Декодирование новостей обратно в текст
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.
items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in
 train_data[0]])

print(reverse_word_index)


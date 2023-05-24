import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Dados de entrada e saída (exemplo)
texts = ['O filme foi ótimo!', 'Não gostei do final...', 'Achei o livro incrível!', 'Que decepção de série...']
labels = [1, 0, 1, 0]  # Sentimentos correspondentes (dados de saída)

# Parâmetros
max_words = 10000  # Número máximo de palavras no vocabulário
max_length = 100  # Comprimento máximo das sequências
embedding_dim = 50  # Dimensão do vetor de embedding
num_filters = 64  # Número de filtros nas camadas convolucionais
hidden_units = 64  # Número de unidades nas camadas densas
num_classes = 2  # Número de classes (sentimentos)

# Validação Cruzada
k = 5  # Número de folds
skf = StratifiedKFold(n_splits=k, shuffle=True)

# Tokenização e vetorização dos textos
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pré-processamento dos dados
X = pad_sequences(sequences, maxlen=max_length)
y = to_categorical(labels, num_classes=num_classes)

# Loop de Validação Cruzada
for train_index, test_index in skf.split(X, np.argmax(y, axis=1)):
    # Divisão dos dados em treinamento e teste para o fold atual
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Construção do modelo
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_length))
    model.add(Conv1D(num_filters, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilação do modelo
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treinamento do modelo
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Avaliação do modelo
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy:', accuracy)

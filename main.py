from model.nn_model import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

if __name__ == "__main__":

    classes = ['koibito', 'doryo']
    observables = ['v_g', 'd', 'h_diff', 'v_diff']
    
    datasets = get_equal_datasets('data/classes/', classes)
    train_ratio = 0.5
    epoch = 100

    train_sets, tests_sets = shuffle_data_set(datasets, train_ratio)

    x_train, y_train = get_data_vectors(train_sets)
    y_binary_train = to_categorical(y_train)

    x_test, y_test = get_data_vectors(train_sets)
    y_binary_test = to_categorical(y_test)
    expected_labels = np.argmax(y_binary_test, axis=1) 

    model = Sequential()

    model.add(Dense(units=200, activation='relu', input_dim=480))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=len(classes), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

    model.fit(x_train, y_binary_train, epochs=epoch, batch_size=32)

    y_predict = model.predict(x_test)
    predicted_labels = np.argmax(y_predict, axis=1) 
    
    cnf_matrix = confusion_matrix(expected_labels, predicted_labels, range(len(classes)))
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                        title='Normalized confusion matrix')

    plt.show()

    

    









    
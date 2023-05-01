import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from colorama import Fore
from keras import Sequential
from keras.applications.densenet import layers
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('./dataset/iris.csv')

del df['Id']

x = df.loc[:, df.columns != 'Species']

enc = OneHotEncoder()
y = df[['Species']]
y = enc.fit_transform(y).toarray()
labels = [label[label.find('_') + 1:] for label in enc.get_feature_names_out()]

x_train, x_valid_test, y_train, y_valid_test = train_test_split(x, y, test_size=0.3)
x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5)

model = Sequential(
    [
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax'),
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=20)

fitted_model = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), epochs=100, callbacks=[es])
loss, accuracy = model.evaluate(x=x_valid, y=y_valid)

model.save('./out/iris.h5')

print(Fore.GREEN + u'\n\u2713 ' + f'Loss ==> {loss}')
print(Fore.GREEN + u'\n\u2713 ' + f'Accuracy ==> {accuracy}')
print(Fore.RESET)

model.summary()

plt.plot(fitted_model.history['accuracy'])
plt.plot(fitted_model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

y_predict = model.predict(x_test)

ax = sns.heatmap(confusion_matrix(y_test.argmax(axis=1), y_predict.argmax(axis=1)), annot=True, cmap='Blues', fmt='g')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.xaxis.set_ticklabels(labels)
ax.yaxis.set_ticklabels(labels)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

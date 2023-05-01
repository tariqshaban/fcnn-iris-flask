import os
import random
import shutil
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Fore
from keras import Sequential
from keras.applications.densenet import layers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

TRANSLATION = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken",
               "gatto": "cat", "mucca": "cow", "pecora": "sheep", "ragno": "spider", "scoiattolo": "squirrel"}

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.2
TEST_SPLIT = 0.1

SOURCE_DIRECTORY = './dataset/animals-10/data/'
REFACTORED_DIRECTORY = './dataset/animals-10/refactored_data/'
TRAIN_DIRECTORY = f'{REFACTORED_DIRECTORY}train/'
VALID_DIRECTORY = f'{REFACTORED_DIRECTORY}valid/'
TEST_DIRECTORY = f'{REFACTORED_DIRECTORY}test/'

if not os.path.exists(REFACTORED_DIRECTORY):
    for label in TRANSLATION.keys():
        os.makedirs(f'{TRAIN_DIRECTORY}{label}', exist_ok=True)
        os.makedirs(f'{VALID_DIRECTORY}{label}', exist_ok=True)
        os.makedirs(f'{TEST_DIRECTORY}{label}', exist_ok=True)

    for label in TRANSLATION.keys():
        numOfFiles = len(next(os.walk(f'{SOURCE_DIRECTORY}{label}/'))[2])

        for image_file in random.sample(glob(f'{SOURCE_DIRECTORY}{label}/*'), int(numOfFiles * TRAIN_SPLIT)):
            shutil.move(image_file, f'{TRAIN_DIRECTORY}{label}')

        for image_file in random.sample(glob(f'{SOURCE_DIRECTORY}{label}/*'), int(numOfFiles * VALID_SPLIT)):
            shutil.move(image_file, f'{VALID_DIRECTORY}{label}')

        for image_file in glob(f'{SOURCE_DIRECTORY}{label}/*'):
            shutil.move(image_file, f'{TEST_DIRECTORY}{label}')

train_batches = ImageDataGenerator().flow_from_directory(
    directory=TRAIN_DIRECTORY,
    classes=TRANSLATION.keys(),
    batch_size=1,
)
valid_batches = ImageDataGenerator().flow_from_directory(
    directory=VALID_DIRECTORY,
    classes=TRANSLATION.keys(),
    batch_size=1,
    shuffle=False,
)
test_batches = ImageDataGenerator().flow_from_directory(
    directory=TEST_DIRECTORY,
    classes=TRANSLATION.keys(),
    batch_size=1,
    shuffle=False,
)

model = Sequential(
    [
        layers.Conv2D(filters=128, kernel_size=3),
        layers.BatchNormalization(),
        layers.Activation(activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=64, kernel_size=3),
        layers.BatchNormalization(),
        layers.Activation(activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(filters=32, kernel_size=3),
        layers.BatchNormalization(),
        layers.Activation(activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=20)

fitted_model = model.fit(x=train_batches, validation_data=valid_batches, epochs=100, callbacks=[es])
loss, accuracy = model.evaluate(x=test_batches)

model.save('./out/animal.h5')

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

y_predict = model.predict(test_batches)

ax = sns.heatmap(confusion_matrix(test_batches.classes, y_predict.argmax(axis=1)), annot=True, cmap='Blues', fmt='g')
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.xaxis.set_ticklabels(TRANSLATION.values())
ax.yaxis.set_ticklabels(TRANSLATION.values())
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

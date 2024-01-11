import numpy as np
from keras.models import Sequential

from keras.layers import Dense,Conv2D, Activation, MaxPool2D, Flatten, Dropout,MaxPooling2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


valid_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size = (32, 32),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size = (32, 32),
                                            batch_size = 1,
                                            class_mode = 'binary')

valid_set = valid_datagen.flow_from_directory('veriler/valid_set',
                                            target_size = (32, 32),
                                            batch_size = 1,
                                            class_mode = 'binary')


model = Sequential()

#1.Katman
model.add(Conv2D(64, 3, data_format='channels_last', kernel_initializer='he_normal', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#2.Katman

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
# %60 unutma işlemi
model.add(Dropout(0.6))

# 3. Katman

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

#4. Katman
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))

# 5. katman

model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
# %60 unutma işlemi
model.add(Dropout(0.6))

### Tam bağlantı Katmanı
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
# %60 unutma işlemi
model.add(Dropout(0.6))

### Çıkış katmanı

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Model Özeti
model.summary()


model.fit(training_set, epochs=10, validation_data=test_set)



"""
pred = model.predict(test_set)

# Extract true labels from the generator
true_labels = []
num_batches = len(test_set)
for i in range(num_batches):
    _, labels = test_set[i]
    true_labels.extend(labels)

true_labels = np.argmax(true_labels, axis=1)
predicted_labels = np.argmax(pred, axis=1)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
conf_matrix = confusion_matrix(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, average='weighted')



print("\n-------------")

print(f"Naive Bayes Doğruluk: {accuracy}")
print(f"Kesinlik: {precision}")
print(f"Duyarlılık: {recall}")
print(f"Karmaşıklık Matrisi:\n{conf_matrix}")
print(f"F1 Skoru: {f1}")

print("-------------\n")
"""
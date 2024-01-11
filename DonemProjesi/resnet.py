
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
import numpy as np

from tensorflow.keras.applications import ResNet50
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
                                                 target_size = (64, 64),
                                                 batch_size = 1,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')

valid_set = valid_datagen.flow_from_directory('veriler/valid_set',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid')) 


for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(training_set, validation_data=valid_set, epochs=1, batch_size=32)

pred = model.predict_generator(test_set,verbose=1)

pred[pred > .5] = 1
pred[pred <= .5] = 0
pred[pred > 1] = 2

test_labels = []

for i in range(0,int(3000)):
    test_labels.extend(np.array(test_set[i][1]))
    

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

accuracy = accuracy_score(test_labels, pred)
precision = precision_score(test_labels, pred, average='weighted')
recall = recall_score(test_labels, pred, average='weighted')
conf_matrix = confusion_matrix(test_labels, pred)
f1 = f1_score(test_labels, pred, average='weighted')


print("\n-------------")

print(f"Naive Bayes Doğruluk: {accuracy}")
print(f"Kesinlik: {precision}")
print(f"Duyarlılık: {recall}")
print(f"Karmaşıklık Matrisi:\n{conf_matrix}")
print(f"F1 Skoru: {f1}")

print("-------------\n")
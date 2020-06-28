import os, glob, numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K

import tensorflow as tf

X_train, X_test, y_train, y_test = np.load('./multi_mushroom_data.npy',allow_pickle=True)
print(X_train.shape)
print(X_train.shape[0])


categories = ["붉은사슴뿔버섯", "새송이버섯", "표고버섯"]
nb_classes = len(categories)

#일반화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255


model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(64, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
    
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

model_dir = './model'
    
if not os.path.exists(model_dir):
    os.mkdir(model_dir)  #학습된 모델이 저장될 폴더 'model'이 없으면 model폴더를 생성한다
    
checkpoint = ModelCheckpoint(filepath=model_dir + '/multi_mushroom_classification.model' , monitor='val_loss', verbose=1, save_best_only=True)
#early_stopping = EarlyStopping(monitor='val_loss', patience=6)

hist = model.fit(
    X_train, 
    y_train, 
    batch_size=16, 
    epochs=50, 
    validation_data=(X_test, y_test), 
    callbacks=[checkpoint]
    )

print()
print("정확도 : %.4f" % (model.evaluate(X_test, y_test)[1]))


y_loss = hist.history['loss']
y_acc = hist.history['accuracy']

y_vloss = hist.history['val_loss']
y_vacc = hist.history['val_accuracy']


x_len = np.arange(len(y_loss))

plt.plot(x_len, y_vloss, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='train_set_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()


plt.plot(x_len, y_vacc, marker='.', c='red', label='val_set_loss')
plt.plot(x_len, y_acc, marker='.', c='green', label='accuracy')
plt.plot(x_len, y_vloss, marker='.', c='blue', label='val_set_acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.show()





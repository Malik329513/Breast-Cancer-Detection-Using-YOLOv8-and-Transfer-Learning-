

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_vgg16_model(input_shape):
    base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_test, y_test, datagen, batch_size=32, epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_test, y_test),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

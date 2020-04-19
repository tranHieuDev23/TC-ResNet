if __name__=='__main__':
    from load_data import load_data_from_folder
    from train import get_tc_resnet_8, get_tc_resnet_14
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint

    X_train, y_train, X_test, y_test, X_validation, y_validation, classes = load_data_from_folder(
        'dataset')
    num_classes = len(classes)
    (num_train, input_length, num_channel) = X_train.shape
    num_test = X_test.shape[0]
    num_validation = X_validation.shape[0]
    
    model_14 = get_tc_resnet_14((input_length, num_channel), num_classes, 1.5)
    model_14.compile(optimizer=Adam(),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint_cb = ModelCheckpoint(
        'weights.{epoch:02d}-{val_loss:.2f}.h5', save_weights_only=True, period=5)
    model_14.fit(x=X_train, y=y_train, batch_size=1024, epochs=30, callbacks=[checkpoint_cb], validation_data=(X_test, y_test))
    model_14.evaluate(X_validation, y_validation)
    model_14.save_weights('weights.h5')
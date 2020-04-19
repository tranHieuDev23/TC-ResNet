from keras import Model
from keras.layers import Input, Conv1D, ReLU, BatchNormalization, Add, AveragePooling1D, Dense, Flatten


def get_residual_block_type_one(input_tensor, c, k):
    multiplied_size = int(c * k)
    x = Conv1D(multiplied_size, 9, strides=1,
               use_bias=False, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(multiplied_size, 9, strides=1,
               use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    x = ReLU()(x)
    return x


def get_residual_block_type_two(input_tensor, c, k):
    multiplied_size = int(c * k)
    x1 = Conv1D(multiplied_size, 9, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)
    x1 = Conv1D(multiplied_size, 9, strides=1,
                use_bias=False, padding='same')(x1)
    x1 = BatchNormalization()(x1)

    x2 = Conv1D(multiplied_size, 1, strides=2,
                use_bias=False, padding='same')(input_tensor)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x = Add()([x1, x2])
    x = ReLU()(x)
    return x


def get_tc_resnet_8(input_shape, num_classes, k):
    input_layer = Input(input_shape)
    x = Conv1D(int(16 * k), 3, strides=1, use_bias=False,
               padding='same')(input_layer)
    x = get_residual_block_type_two(x, 24, k)
    x = get_residual_block_type_two(x, 32, k)
    x = get_residual_block_type_two(x, 48, k)
    x = AveragePooling1D(3, 1)(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)


def get_tc_resnet_14(input_shape, num_classes, k):
    input_layer = Input(input_shape)
    x = Conv1D(int(16 * k), 3, strides=1, use_bias=False,
               padding='same')(input_layer)
    x = get_residual_block_type_two(x, 24, k)
    x = get_residual_block_type_one(x, 24, k)
    x = get_residual_block_type_two(x, 32, k)
    x = get_residual_block_type_one(x, 32, k)
    x = get_residual_block_type_two(x, 48, k)
    x = get_residual_block_type_one(x, 48, k)
    x = AveragePooling1D(3, 1)(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

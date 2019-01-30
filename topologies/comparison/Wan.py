from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional, LeakyReLU, concatenate, BatchNormalization, PReLU
from keras.optimizers import RMSprop


MODEL_CONV_FILTERS = 32
MODEL_CONV_KERNEL_SIZE = 18
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'


def build_model(input_shape, appliances):
    seq_length = input_shape[0]

    x = Input(shape=input_shape)

    conv_1 = Conv1D(filters=32, kernel_size=3, padding=MODEL_CONV_PADDING, activation='linear')(x)
    flat_5 = Flatten()(conv_1)
#===============================================================================================
    conv_10 = Conv1D(filters=32, kernel_size=5, padding=MODEL_CONV_PADDING, activation='linear')(x)
    flat_50 = Flatten()(conv_10)
#===============================================================================================
    conv_11 = Conv1D(filters=32, kernel_size=7, padding=MODEL_CONV_PADDING, activation='linear')(x)
    flat_51 = Flatten()(conv_11)
#===============================================================================================
    # merge
    concate_5 = concatenate([flat_5, flat_50, flat_51])
    reshape_8 = Reshape(target_shape=(seq_length, -1))(concate_5)
    outputs_disaggregation = []

    for appliance_name in appliances:
        biLSTM_1 = Bidirectional(LSTM(128, return_sequences=True))(reshape_8)
        biLSTM_2 = Bidirectional(LSTM(128, return_sequences=True))(biLSTM_1)
        biLSTM_2 = Bidirectional(LSTM(128, return_sequences=True))(biLSTM_1)
        FC_1 = TimeDistributed(Dense(128, activation='relu'))(biLSTM_2)
        outputs_disaggregation.append(Reshape(target_shape=(seq_length,1), name=appliance_name.replace(" ", "_"))(TimeDistributed(Dense(1, activation='linear'))(FC_1)))

    model = Model(inputs=x, outputs=outputs_disaggregation)
    optimizer = RMSprop(lr=0.001, clipnorm=40)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

    return model

""" Denoising Auto-Encoder """
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D
from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional


MODEL_CONV_FILTERS = 16
MODEL_CONV_KERNEL_SIZE = 4
MODEL_CONV_STRIDES = 1
MODEL_CONV_PADDING = 'same'

## cnn*2 + dense*3(128) 

def build_model(input_shape, appliances):
    seq_length = input_shape[0]

    # build it!
    x = Input(shape=input_shape)
    # conv
    conv_1 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING,activation='linear')(x)
    # reshape
    conv_1 = Flatten()(conv_1)
    conv_1 = Reshape(target_shape=(seq_length, -1))(conv_1)
    # dense

    # Initialization
    outputs_disaggregation = []
    for appliance in appliances:
        biLSTM_1 = Bidirectional(LSTM(128, return_sequences=True))(conv_1)
        biLSTM_2 = Bidirectional(LSTM(256, return_sequences=True))(biLSTM_1)
        FC_1 = TimeDistributed(Dense(128, activation='tanh'))(biLSTM_2)
        outputs_disaggregation.append(Reshape(target_shape=(seq_length,1))(TimeDistributed(Dense(1, activation='linear'))(FC_1)))


    # compile it!
    model = Model(inputs=x, outputs=outputs_disaggregation)
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    return model

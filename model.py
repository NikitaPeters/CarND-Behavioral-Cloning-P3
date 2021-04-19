import tensorflow.contrib.keras as keras
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Activation, Cropping2D, Dropout
import utilities

def nvidia_model(optimizer, loss='mse'):
    """
    Rebuild of the convolution neural network from M. Bojarski et al. from NVIDIA 2016-04-25
    
    The network consists of 9 layers:1 normalization, 5 convolutional layers and 3 fully connected layers.
    I used the exponential linear unit (ELU) activation instead of a simple rectified linear unit (ReLU), 
    because the area of nonlinearity is higher and I get better results. For the last 2 I use ReLus.
    """
    model = Sequential()

    #model.add(Lambda(lambda x: x /127.5 - 1.0, input_shape = (utilities.image_h, utilities.image_w, utilities.image_d)))
    model.add(Lambda(lambda x: x /255 - 0.5, input_shape = (utilities.image_h, utilities.image_w, utilities.image_d)))
   
    #5 convolutional layers
    # strided convolutions in the first 3 Conv-layers with a 2x2 stride and a 5x5 kernel
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    # non strided convolutions in the last 2 Conv-layers with a 2x2 stride and a 3x3 kernel
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))

    model.add(Flatten())
    #Dropout to reduce overfitting
    model.add(Dropout(0.5))
    
    # 3 fully connected layer
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model
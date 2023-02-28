# import necessary packages 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class KiNet_mlp:


    def create_mlp(dim, regress = False):

	# define our MLP Network: 1000 - 500 architecture

        model = Sequential()
        
        model.add(Dense(1000, input_dim = dim, activation = "relu"))  
        
        
        model.add(Dense(500, activation = "relu"))

    	# check to see if the regression node is to be added

        if regress: # if we are performing regression, we add a Dense layer containing a single neuron with a linear activation function
            model.add(Dense(1, activation = "linear"))

    	# return our model
        
        # return the mlp model
        model.summary()
        return model	   


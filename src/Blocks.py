# This is directly taken from https://github.com/Fzaero/BRDPN/blob/master/src/Blocks.py
from keras.layers.core import Layer
from keras.models import Sequential,Model
from keras import regularizers
from keras.layers import Input,Dense,Multiply,Concatenate,LSTM,TimeDistributed,Permute,Dropout,Activation,CuDNNLSTM,Dropout,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras import backend as K
import keras
from keras.layers.normalization import BatchNormalization
regul=0.001 

# These models are used in Networks.py file for creating Multilayer Perceptrons used in the Graph Neural Network
class RelationalModel(Layer):

    def __init__(self, input_size,n_of_features,filters,rm=None,reuse_model=False,**kwargs):
        self.input_size=input_size
        self.n_of_features=n_of_features
        n_of_filters = len(filters)
        if(reuse_model):
            relnet = rm    
        else:
            input1 = Input(shape=(n_of_features,))
            x=input1
            for i in range(n_of_filters-1):
                x = Dense(filters[i],kernel_regularizer=regularizers.l2(regul),activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='relu')(x)
                
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
              bias_regularizer=regularizers.l2(regul),activation='linear')(x)    
            relnet = Model(inputs=[input1],outputs=[x])
        self.relnet = relnet
        self.output_size = filters[-1]
        
        super(RelationalModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.relnet.build((None,self.n_of_features)) #,self.input_size
        self.trainable_weights = self.relnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,)+input_size+(int(output_size),)

    def call(self, X):
        X = K.reshape(X,(-1,self.n_of_features))
        output = self.relnet.call(X)
        output= K.reshape(output,((-1,)+self.input_size+(self.output_size,)))
        return output
    
    def getRelnet(self):
        return self.relnet


class ObjectModel(Layer):
    
    def __init__(self, input_size,n_of_features,filters,om=None,reuse_model=False,**kwargs):
        self.input_size=input_size 
        self.n_of_features=n_of_features
        n_of_filters = len(filters)

        if (reuse_model):
            objnet = om
        else:
            input1 = Input(shape=(n_of_features,))
            x=input1
            for i in range(n_of_filters-1):
                x = Dense(filters[i],kernel_regularizer=regularizers.l2(regul),activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul),activation='relu')(x)
            x = Dense(filters[-1],kernel_regularizer=regularizers.l2(regul),
                  bias_regularizer=regularizers.l2(regul),activation='linear')(x)
                
            objnet = Model(inputs=[input1],outputs=[x])
        self.objnet = objnet

        self.output_size = filters[-1]
        super(ObjectModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.objnet.build((None,self.input_size,self.n_of_features))
        self.trainable_weights = self.objnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size

        return (None,)+input_size+(int(output_size),)

    def call(self, X):
        X = K.reshape(X,(-1,self.n_of_features))
        output = self.objnet.call(X)
        output= K.reshape(output,((-1,)+self.input_size+(self.output_size,)))
        return output

    def getObjnet(self):
        return self.objnet


# model to be used to classify the whole graph
# filters = [feature_size, # of classes]
class ClassificationModel(Layer):
    def __init__(self, input_size, n_of_features, filters, cm=None, reuse_model=False, n_of_frame=150, **kwargs):
        self.input_size=input_size
        self.n_of_features=n_of_features
        self.n_of_frame=n_of_frame
        if(reuse_model):
            classnet = cm    
        else:
            input1 = Input(shape=(n_of_frame,n_of_features)) 
            x = LSTM(filters[0],kernel_regularizer=regularizers.l2(regul), # lstm to remember all of the frames
                        bias_regularizer=regularizers.l2(regul))(input1)
            x = Dense(filters[1],kernel_regularizer=regularizers.l2(regul), # dense to classify the whole graph
                                bias_regularizer=regularizers.l2(regul),activation='softmax')(x)
            classnet = Model(inputs=[input1],outputs=[x])
            self.classnet = classnet
            self.output_size = filters[-1]
        
        super(ClassificationModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.classnet.build((None,self.n_of_features)) #,self.input_size
        self.trainable_weights = self.classnet.trainable_weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,input_size,int(output_size))

    def call(self, X):              
        output = self.classnet.call(X)
        return output
    
    def getClassnet(self):
        return self.classnet
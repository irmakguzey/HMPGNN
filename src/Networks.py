from keras.layers import Permute,Subtract,Add,Lambda,Input,Concatenate,TimeDistributed,Activation,Dropout,dot,Reshape
import tensorflow as tf
from keras.activations import tanh,relu
from keras import optimizers
from keras import regularizers
from Blocks import *

import json
import numpy as np

class PropagationNetwork:
    def __init__(self):
        self.Nets={}
        self.set_weights = False # This is for reusing the model (i guess)
    def getModel(self, n_of_frame, n_objects, n_of_relations, n_of_class, n_of_features, num_object_attr=2, num_relation_attr=2):
        if n_objects in self.Nets.keys():
            return self.Nets[n_objects]

        objects = Input(shape=(n_of_frame, n_objects, num_object_attr), name='objects')
        relations = Input(shape=(n_of_frame, n_of_relations, num_relation_attr), name='relations')
        sender_flags = Input(shape=(n_of_frame, n_objects, n_of_relations), name='sender_flags') # flag representing whether indexed object participate in the indexed relation
        receiver_flags = Input(shape=(n_of_frame, n_objects, n_of_relations), name='receiver_flags')
        propagation = Input(shape=(n_of_frame, n_objects, n_of_features), name='propagation')

        print('objects.shape: {}, relations.shape: {}, sender/receiver_flags.shape: {}, propagation,shape: {}'.format(objects.shape,
                                                                                                                      relations.shape,
                                                                                                                      sender_flags.shape,
                                                                                                                      propagation.shape))

        # change the columns 2 and 3 to fix the matrix multiplication in matmul layer prop_senders and prop_receivers
        permuted_sender_flags = Permute((1,3,2))(sender_flags)
        permuted_receiver_flags = Permute((1,3,2))(receiver_flags)

        # Using the same variables as @Fzaero's BRDPN depository
        if(self.set_weights):
            # encoder models
            rm = RelationalModel((n_of_frame, n_of_relations,),2,[150,150,150,150],self.relnet,True)
            om = ObjectModel((n_of_frame, n_objects,),2,[100,100],self.objnet,True)

            # propagation models
            rmp = RelationalModel((n_of_frame, n_of_relations,),350,[150,150,n_of_features],self.relnetp,True)
            omp = ObjectModel((n_of_frame, n_objects,),300,[100,n_of_features],self.objnetp,True)
            
            # output model
            cm = ClassificationModel((n_of_frame, n_of_features,), n_of_features, [100, n_of_class], n_of_frame=n_of_frame)
        else:
            # encoder models
            rm=RelationalModel((n_of_frame, n_of_relations,),2,[150,150,150,150])
            om = ObjectModel((n_of_frame, n_objects,),2,[100,100])
            
            # propagation models
            rmp=RelationalModel((n_of_frame, n_of_relations,),350,[150,150,n_of_features])
            omp = ObjectModel((n_of_frame, n_objects,),300,[100,n_of_features])

            # output model
            cm = ClassificationModel((n_of_frame, n_of_features,), n_of_features, [100, n_of_class], n_of_frame=n_of_frame)
            
            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()
        
        rel_encoding=Activation('relu')(rm(relations))
        obj_encoding=Activation('relu')(om(objects))
        rel_encoding=Dropout(0.1)(rel_encoding)
        obj_encoding=Dropout(0.1)(obj_encoding)
        prop=propagation

        # matmul layers
        prop_senders = Lambda(lambda x:tf.matmul(permuted_sender_flags, x), output_shape=(n_of_frame, n_of_relations, 100))
        prop_receivers = Lambda(lambda x:tf.matmul(permuted_receiver_flags, x), output_shape=(n_of_frame, n_of_relations, 100))
        prop_objects = Lambda(lambda x:tf.matmul(receiver_flags, x), output_shape=(n_of_frame, n_objects, 100))

        # Creating the propagation
        for _ in range(5):
            senders_prop = prop_senders(prop)
            receivers_prop = prop_receivers(prop)
            rmp_vector = Concatenate()([rel_encoding, senders_prop, receivers_prop])
            rmp_outcome = Activation('relu')(rmp(rmp_vector))
            effect_receivers = prop_objects(rmp_outcome)
            omp_vector = Concatenate()([obj_encoding, effect_receivers, prop])
            prop = Activation('relu')(omp(omp_vector))

        print('propagation done')
        print('rmp_vector.shape: {}, rmp_outcome.shape: {}, omp_vector.shape: {}'.format(rmp_vector.shape, rmp_outcome.shape, omp_vector.shape))
        print('prop.shape: {}'.format(prop.shape)) # prop.shape: ((n_of_frame, n_object, n_of_features))

        # take the mean of prop vector in axis for objects - prepare the output for classification model
        get_prop_mean = Lambda(lambda x:tf.reduce_mean(x, 2))
        mean_prop = get_prop_mean(prop)
        print('mean_prop.shape: {}'.format(mean_prop.shape))

        # put them to classification model
        predicted_classes = cm(mean_prop)
        predicted_classes = Lambda(lambda x: x[:,:], output_shape=(n_of_class,), name='target')(predicted_classes)
        print('predicted_classes.shape: {}'.format(predicted_classes.shape))

        model = Model(inputs=[objects, relations, sender_flags, receiver_flags, propagation], outputs=[predicted_classes])       
        adam = optimizers.Adam(lr=0.0005, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['binary_accuracy'])
        self.Nets[n_objects]=model
        return model

class PredictionPropNet:
    def __init__(self):
        self.Nets = {}
        self.set_weights = False
    
    # n_of_desired_attr: number of dimensions to be guessed, the first n_of_desired_attr is learned and predicted
    def get_model(self, n_of_frames, n_of_points, n_of_relations, n_of_features=100 ,n_of_object_attr=4, n_of_relation_attr=2, n_of_desired_attr=2): 
        if n_of_points in self.Nets.keys():
            return self.Nets[n_of_points]

        objects = Input(shape=(n_of_points, n_of_object_attr), name='objects')
        relations = Input(shape=(n_of_relations, n_of_relation_attr), name='relations')
        sender_flags = Input(shape=(n_of_points, n_of_relations), name='sender_flags')
        receiver_flags = Input(shape=(n_of_points, n_of_relations), name='receiver_flags')
        control_flags = Input(shape=(n_of_points, n_of_desired_attr), name='control_flags')
        propagation = Input(shape=(n_of_points, n_of_features), name='propagation')
        
        print('objects.shape: {}, relations.shape: {}, sender/receiver_flags.shape: {}, propagation,shape: {}'.format(objects.shape,
                                                                                                                      relations.shape,
                                                                                                                      sender_flags.shape,
                                                                                                                      propagation.shape))

        # change the columns 2 and 3 to fix the matrix multiplication in matmul layer prop_senders and prop_receivers
        permuted_sender_flags = Permute((2,1))(sender_flags)
        permuted_receiver_flags = Permute((2,1))(receiver_flags)

        # # Using the same variables as @Fzaero's BRDPN depository
        if(self.set_weights):
            # encoder models
            rm = RelationalModel((n_of_relations,),n_of_relation_attr,[150,150,150,150],self.relnet,True)
            om = ObjectModel((n_of_points,),n_of_object_attr,[100,100],self.objnet,True)

            # propagation models
            rmp = RelationalModel((n_of_relations,),350,[150,150,100],self.relnetp,True)
            omp = ObjectModel((n_of_points,),300,[100,n_of_features+n_of_object_attr],self.objnetp,True) # n_of_features+2 -> n_of_features for giving the output back to propagation, 2 is for guessing the output
            
        else:
            # encoder models
            rm=RelationalModel((n_of_relations,),n_of_relation_attr,[150,150,150,150])
            om = ObjectModel((n_of_points,),n_of_object_attr,[100,100])
            
            # propagation models
            rmp=RelationalModel((n_of_relations,),350,[150,150,100])
            omp = ObjectModel((n_of_points,),300,[100,n_of_features+n_of_object_attr])

            self.set_weights=True
            self.relnet=rm.getRelnet()
            self.objnet=om.getObjnet()
            self.relnetp=rmp.getRelnet()
            self.objnetp=omp.getObjnet()
        
        rel_encoding=Activation('relu')(rm(relations))
        obj_encoding=Activation('relu')(om(objects))
        rel_encoding=Dropout(0.1)(rel_encoding)
        obj_encoding=Dropout(0.1)(obj_encoding)
        prop=propagation

        # matmul layers
        prop_senders = Lambda(lambda x:tf.matmul(permuted_sender_flags, x), output_shape=(n_of_relations, n_of_features))
        prop_receivers = Lambda(lambda x:tf.matmul(permuted_receiver_flags, x), output_shape=(n_of_relations, n_of_features))
        prop_objects = Lambda(lambda x:tf.matmul(receiver_flags, x), output_shape=(n_of_points, n_of_features))
        prop_layer = Lambda(lambda x: x[:,:,n_of_object_attr:], output_shape=(n_of_points, n_of_features)) # prop_layer is used for getting the propagation from the object neural network

        # have the propagation between networks
        for _ in range(10):
            senders_prop = prop_senders(prop)
            receivers_prop = prop_receivers(prop)
            rmp_vector = Concatenate()([rel_encoding, senders_prop, receivers_prop])
            rmp_outcome = rmp(rmp_vector)
            effect_receivers = Activation('tanh')(prop_objects(rmp_outcome))
            omp_vector = Concatenate()([obj_encoding, effect_receivers, prop])
            omp_outcome = omp(omp_vector)
            prop = Activation('tanh')(Add()([prop_layer(omp_outcome), prop]))

        print('prop.shape: {}'.format(prop.shape)) # prop.shape: ((n_of_frame, n_object, n_of_features))

        # take the 4 dimensions, hidden in the matrix prop
        take_predictions = Lambda(lambda x: Activation('tanh')(x[:,:,:n_of_desired_attr]), output_shape=(n_of_points, n_of_desired_attr))
        control_predictions = Lambda(lambda x: Multiply()([x, control_flags]), output_shape=(n_of_points, n_of_desired_attr), name='target')
        predicted = control_predictions(take_predictions(omp_outcome))
        print('predicted.shape: {}'.format(predicted.shape))

        model = Model(inputs=[objects, relations, sender_flags, receiver_flags, control_flags, propagation], outputs=[predicted])       
        adam = optimizers.Adam(lr=0.00005, decay=0.0)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
        
        self.Nets[n_of_points]=model
        return model
 


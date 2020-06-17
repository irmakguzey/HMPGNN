import c3d
import h5py
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import operator
import os
import random
import statistics as stats
import sys

from keras.models import model_from_json
from matplotlib.animation import FuncAnimation
from nltk.tag import pos_tag, map_tag
from nltk.stem.lancaster import LancasterStemmer

from Networks import *
from Blocks import *

# This class makes predictions for the joints of the code, given a trajectory with missing joints
class HumanMotionPredictor:
    
    def __init__(self, max_num_moves, max_file_digit, dir_path, data_path, n_of_points, n_of_frames=1500):
        self.max_num_moves = max_num_moves
        self.max_file_digit = max_file_digit
        self.dir_path = dir_path
        self.data_path = data_path # directory to save the arrays
        self.init_n_of_points = n_of_points
        self.n_of_frames = n_of_frames

        self.scaling_constant = 1000
        self.scaling_constants = [200,1000] # TODO: check this value out, this is for scaling the distances between 0-1 in the gnn_model 200 for x, 1000 is for y
        
        self.object_dim = 3
        self.n_of_object_attr = 3 # n_of_object_attr is 2 for preprocessing data, but when trained 
        self.n_of_training_attr = 6 # n_of_object_attr is 2 for preprocessing data, but for training controlled indices' velocities are also given 
        self.n_of_relation_attr = 3

        # create hardcoded indices for relations and points        
        # NOTE: indices for each necessary point is as follows:
        # 2: stomach
        # 4,6: chest to stomach
        # 9,10,11,13: left arm: shoulder to hand
        # 19,20,21,23: right arm: shoulder to hand
        # 31,32: head
        # 33,37,39,41: left hip and leg
        # 34,45,47,49: right hip and leg
        self.wanted_indices = [2, 4,6, 9,10,11,13, 19,20,21,23, 31,32,  33,37,39,41, 34,45,47,49]
        self.base_point_index = 2 # stomach will be taken as the base for the distance to be put (instead of average)

        # NOTE: relations are hard coded according to the wanted_indices
        # when it is used, since wanted_indices are taken from the data and put to points as scaled, wanted_indices should be used once more while creating relations
        # such as, when encoding is created actual index should be found as self.wanted_indices.index(element_in_relations)
        self.hardcoded_relations = {
            31: [32,6], 32: [31,6], # head's connection with the chest
            6: [31,32,4,9,19], # chest's connection with the head, shoulders and the stomach
            4: [2,6], # stomach's connection with the chest and the groin
            2: [4,33,34], # groin's connection with the stomach and the hips
            9: [6,10], 10: [9,11], 11: [10,13], 13: [11], # left arm connections
            19: [6,20], 20: [19,21], 21: [20,23], 23: [21], # right arm connections
            33: [2,37], 37: [33,39], 39: [37,41], 41: [39], # left leg
            34: [2,45], 45: [34,47], 47: [45,49], 49: [47] # right leg
        }
        n_of_relations = 0
        self.actual_hardcoded_relations = [] # this is the actual indices of the indices above (when got from the data, these points' indices will change)
        for i,num in enumerate(self.wanted_indices):
            self.actual_hardcoded_relations.append([])
            for j in self.hardcoded_relations[num]:
                self.actual_hardcoded_relations[i].append(self.wanted_indices.index(j))
            n_of_relations += len(self.actual_hardcoded_relations[i])
        
        # update n_of_points and n_of_relations according to the hardcoded points and relations
        if os.path.exists(self.data_path+'points.npy'):
            points = np.load(open(self.data_path+'points.npy', 'rb'))
            self.n_of_traj = points.shape[0]
        self.n_of_relations = n_of_relations
        self.n_of_points = len(self.wanted_indices)

        # set the control flags - this array will indicate which joints will be given as input and which ones will not
        control_indices = [2, 4,6, 9,10,11, 19,20,21,  33,37,39, 34,45,47] # these indices will NOT be given
        self.actual_control_indices = [self.wanted_indices.index(ci) for ci in control_indices]
        self.control_flags = np.zeros(self.n_of_points)
        for ci in control_indices:
            actual_index_ci = self.wanted_indices.index(ci)
            self.control_flags[actual_index_ci] = 1


    def preprocess_data(self):

        if os.path.exists(self.data_path + 'points.npy') and os.path.exists(self.data_path + 'base_point.npy'):
            points = np.load(open(self.data_path + 'points.npy', 'rb'))
            base_point = np.load(open(self.data_path + 'base_point.npy', 'rb'))
            self.n_of_traj = points.shape[0]
            print('points array is loaded')
        else:
            # create the readers to receive the data
            readers = []
            reader_files = []
            for move_num in range(self.max_num_moves+1):
                str_move_num = str(move_num)
                while len(str_move_num) < self.max_file_digit:
                    str_move_num = '0' + str_move_num
                file_path = self.dir_path + str_move_num + '_raw.c3d'
                if os.path.exists(file_path):
                    reader_file = open(file_path, 'rb')
                    reader_files.append(reader_file)
                    readers.append(c3d.Reader(reader_file))

            # calculate the number of eligible trajectories with the given points number
            # and remove the readers with trajectories that are different then the given number of points
            point_counts = 0
            copy_readers = [r for r in readers]
            for t in range(len(copy_readers)):
                for _,data_pts,_ in copy_readers[t].read_frames():
                    points_num = data_pts.shape[0]
                    if points_num == self.init_n_of_points:
                        point_counts += 1
                    else:
                        readers.remove(copy_readers[t])
                    break
            self.n_of_traj = point_counts # number of trajectory with the given n_of_points

            # create the points array
            points = np.zeros((self.n_of_traj, self.n_of_frames, self.n_of_points, self.object_dim))
            base_point = np.zeros((self.n_of_traj, self.n_of_frames, 1, self.object_dim)) # this array holds the positions of the base point
            for t in range(self.n_of_traj):

                # ignore larger frames in points 
                last_frame = readers[t].last_frame()
                first_frame = readers[t].first_frame()
                points[t,(last_frame-first_frame):,:,:] = np.nan # put nan to the parts where there are not enough frames in a trajectory
                base_point[t, (last_frame-first_frame):,:,:] = np.nan

                for f,data_pts,_ in readers[t].read_frames():

                    if f-first_frame >= self.n_of_frames: # don't take the frames larger than n_of_frames
                        break
                    
                    # put the data into points
                    points[t,f-first_frame,:,:] = data_pts[self.wanted_indices,:self.object_dim]
                    base_point[t,f-first_frame,:,:] = data_pts[self.base_point_index,:self.object_dim]

            # close the reader_files
            for r in reader_files:
                r.close()
            print('points and base_point array is saved')
            np.save(open(self.data_path + 'points.npy', 'wb'), points)
            np.save(open(self.data_path + 'base_point.npy', 'wb'), base_point)

        if os.path.exists(self.data_path+'objects.npy') and os.path.exists(self.data_path+'objects_flag.npy'):
            objects = np.load(open(self.data_path+'objects.npy', 'rb'))
            objects_flag = np.load(open(self.data_path+'objects_flag.npy', 'rb'))
            print('objects and objectds_flag loaded')
        else:
            # create the objects and objects_flag arrays - difference between the points and the objects is that objects is not biased
            objects = np.zeros((self.n_of_traj, self.n_of_frames, self.n_of_points, self.n_of_object_attr))
            # objects[:,-1,:,2:] = 0 # TODO: do this if velocity is needed, set the velocity of all objects at the end to zero
            objects_flag = np.zeros((self.n_of_traj, self.n_of_frames, self.n_of_points, 1))
            for t in range(self.n_of_traj):
                for f in range(self.n_of_frames):
                    if np.isnan(points[t,f,0,0]) or np.isnan(points[t,f,0,1]):
                        break
                    # x_mean = np.nanmean(points[t,f,:,0])
                    # y_mean = np.nanmean(points[t,f,:,1])
                    # # put the positions
                    # objects[t,f,:,0] = points[t,f,:,0] - x_mean
                    # objects[t,f,:,1] = points[t,f,:,1] - y_mean
                    objects[t,f,:,:] = points[t,f,:,:] - base_point[t,f,0,:]
                    # put the velocities TODO: put the velocity if necessary
                    # if f < self.n_of_frames-1:
                    #     objects[t,f,:,2:] = objects[t,f+1,:,:2] - objects[t,f,:,:2]
                    objects_flag[t,f,:,0] = 1
            np.save(open(self.data_path+'objects.npy','wb'), objects)
            np.save(open(self.data_path+'objects_flag.npy','wb'), objects_flag)
            print('objects and objects_flag arrays are saved')
        
        if os.path.exists(self.data_path+'relations.npy') and os.path.exists(self.data_path+'senders.npy') and os.path.exists(self.data_path+'receivers.npy'):
            relations = np.load(open(self.data_path+'relations.npy', 'rb'))
            senders = np.load(open(self.data_path+'senders.npy', 'rb'))
            receivers = np.load(open(self.data_path+'receivers.npy', 'rb'))
            print('relations, senders and receivers are loaded')
        else:
            # create the relations array - this array is created from the self.hardcoded_relations array
            # senders and receivers will be the same for every frame and trajectory
            relations = np.zeros((self.n_of_traj, self.n_of_frames, self.n_of_relations, self.n_of_relation_attr))
            senders = np.zeros((self.n_of_points, self.n_of_relations)) # senders and relations are the same for all frames and trajectories
            receivers = np.zeros((self.n_of_points, self.n_of_relations))
            sender_set = receiver_set = False
            relation_index = 0
            for t in range(self.n_of_traj):
                for f in range(self.n_of_frames):
                    if objects_flag[t,f,0,0] == 0:
                            break
                    # traverse in the relation array and put the distance between the senders and relations to the relation array
                    for sender_index in range(len(self.actual_hardcoded_relations)):
                        # set sender flag
                        if not sender_set: 
                            senders[sender_index, relation_index] = 1
                            sender_set = True
                        
                        for receiver_index in self.actual_hardcoded_relations[sender_index]:
                            # set the receiver flag
                            if not receiver_set:
                                receivers[receiver_index, relation_index] = 1
                                receiver_set = True

                            # put the relation distance into the relation array
                            relations[t,f,relation_index,:] = points[t,f,sender_index,:] - points[t,f,receiver_index,:]
                            
                            # update the relation index
                            relation_index += 1
                    
                    # reset the relation_index 
                    relation_index = 0

            np.save(open(self.data_path+'relations.npy','wb'), relations)
            np.save(open(self.data_path+'senders.npy','wb'), senders)
            np.save(open(self.data_path+'receivers.npy','wb'), receivers)
            print('relations, senders and receivers are saved')

    # train the graph neural network and save the weights afterwards
    def train_gnn(self, chain_effects=True, indices_for_train=None, show_error=False):

        # load the data
        objects = np.load(open(self.data_path+'objects.npy','rb')) # this one loads all of the trajectories
        relations = np.load(open(self.data_path+'relations.npy','rb'))
        sender_flags = np.load(open(self.data_path+'senders.npy','rb'))
        receiver_flags = np.load(open(self.data_path+'receivers.npy', 'rb'))
        objects_flag = np.load(open(self.data_path+'objects_flag.npy', 'rb'))

        # scale the array values between -1,1 to have accurate training
        objects = objects / self.scaling_constant
        relations = relations / self.scaling_constant

        # create the propagation array 
        n_of_features = 100
        propagation = np.zeros((self.n_of_traj, self.n_of_points, n_of_features))

        # calculate the positions of the points in the next frame
        target = np.zeros((self.n_of_traj, self.n_of_frames-1, self.n_of_points, self.n_of_object_attr))
        target[:,:,:,:] = objects[:,1:,:,:self.n_of_object_attr]
        print('target.shape: {}'.format(target.shape))

        # get the gnn model
        prop_net = PredictionPropNet()
        self.gnn_model = prop_net.get_model(n_of_frames=self.n_of_frames, n_of_points=self.n_of_points, n_of_relations=self.n_of_relations, n_of_features=n_of_features,
                                            n_of_object_attr=self.n_of_training_attr, n_of_relation_attr=self.n_of_relation_attr, n_of_desired_attr=self.n_of_object_attr) 

        # set the trajectory indices that will be given for training
        # indices_for_train = [199, 1356, 518, 40, 1123] # NOTE: these indices are put hardcoded
        if indices_for_train == None:
            indices_for_train = np.arange(self.n_of_traj)
        n_of_train_traj = len(indices_for_train)

        # expand senders and receivers (since senders and receivers are the same for all trajectories and frames)
        expanded_senders = np.repeat(np.expand_dims(sender_flags, axis=0), n_of_train_traj, axis=0)
        expanded_receivers = np.repeat(np.expand_dims(receiver_flags, axis=0), n_of_train_traj, axis=0)
        expanded_control_flag = np.repeat(np.expand_dims(np.repeat(np.expand_dims(self.control_flags, axis=1), self.n_of_object_attr, axis=1), axis=0), n_of_train_traj, axis=0) # expand control_flag (21,) -> (traj_num, 21, obj_attr,)

        prediction_objects = objects[indices_for_train,0,:,:]
        controlled_predictions = objects[indices_for_train,0,:,:] * expanded_control_flag # positions of non-given objects
        predicted_objects = np.zeros((n_of_train_traj, self.n_of_frames-1, self.n_of_points, self.n_of_object_attr)) 
        training_errors = []
        # train the model for each of the frames
        for f in range(self.n_of_frames-1):
            if np.max(objects_flag[indices_for_train,f+1,:,:]) == 0: # break if all of the trajectories' frames are finished
                break
            
            print('frame_num: {}'.format(f))

            # prepare objects arrays to expand them to have next positions for non-given objects
            propagation = np.zeros((self.n_of_traj, self.n_of_points, n_of_features))
            controlled_target = target[indices_for_train,f,:,:] * expanded_control_flag # zero for given objects, next positions for non-given objects
            if chain_effects == False: # the predictions are not used as object attributes in the next steps
                training_objects = np.concatenate((objects[indices_for_train,f,:,:], controlled_target - controlled_predictions), axis=2)
            else:
                training_objects = np.concatenate((prediction_objects, controlled_target - controlled_predictions), axis=2)
            
            self.gnn_model.fit({'objects': training_objects,
                                'relations': relations[indices_for_train,f,:,:],
                                'sender_flags': expanded_senders,
                                'receiver_flags': expanded_receivers,
                                'control_flags': expanded_control_flag,
                                'propagation': propagation[indices_for_train,:,:]},
                                {'target': controlled_target},
                                shuffle=True,
                                epochs = 10,
                                verbose = 1)


            predictions = self.gnn_model.predict({'objects': training_objects,
                                                  'relations': relations[indices_for_train,f,:,:],
                                                  'sender_flags': expanded_senders,
                                                  'receiver_flags': expanded_receivers,
                                                  'control_flags': expanded_control_flag,
                                                  'propagation': propagation[indices_for_train,:,:]})


            # change the batch that will be given for the next prediction by using control flag
            controlled_predictions = predictions * expanded_control_flag # have the joints that are not given, put zeros for the joints that are given   
            controlled_given_objects = target[indices_for_train,f,:,:] * -(expanded_control_flag-1) # have the joints that are given, put zeros for the joints that are not given
            prediction_objects = controlled_predictions + controlled_given_objects # have them summed elementwise

            # put the prediction in this frame to the predicted_objects
            predicted_objects[:,f,:,:] = prediction_objects[:,:,:] * self.scaling_constant

            # calculate the error in the current frame
            current_error = np.sum(np.absolute(predicted_objects[:,f,:,:] - target[indices_for_train,f,:,:]*self.scaling_constant))
            training_errors.append(current_error)
            print('error in frame {} is: {}'.format(f, current_error))

        # plot the error
        if show_error:
            plt.figure()
            plt.plot(range(len(training_errors)), training_errors)
            plt.title('Training Errors')
            plt.xlabel('Frame')
            plt.ylabel('Error')
            plt.show()

        # save gnn_model
        self.save_gnn()

        return predicted_objects, training_errors

    # make autoregressive predictions for every frame
    def make_predictions(self, chain_effects=True, indices_for_predict=None, show_error=False):

        # load the data
        objects = np.load(open(self.data_path+'objects.npy','rb')) # this one loads all of the trajectories
        relations = np.load(open(self.data_path+'relations.npy','rb'))
        sender_flags = np.load(open(self.data_path+'senders.npy','rb'))
        receiver_flags = np.load(open(self.data_path+'receivers.npy', 'rb'))
        objects_flag = np.load(open(self.data_path+'objects_flag.npy', 'rb'))

        # scale the arrays between -1 and 1 for accurate training
        objects = objects / self.scaling_constant
        relations = relations / self.scaling_constant

        # create the propagation array 
        n_of_features = 100
        propagation = np.zeros((self.n_of_traj, self.n_of_points, n_of_features))

        # calculate the positions of the points in the next frame
        target = np.zeros((self.n_of_traj, self.n_of_frames-1, self.n_of_points, self.n_of_object_attr))
        target[:,:,:,:] = objects[:,1:,:,:self.n_of_object_attr]
        print('target.shape: {}'.format(target.shape))

        # get the gnn model
        self.load_gnn()

        # set the trajectory indices that will be given for training
        # indices_for_train = [199, 1356, 518, 40, 1123] # NOTE: these indices are put hardcoded
        if indices_for_predict == None:
            indices_for_predict = np.arange(self.n_of_traj)
        n_of_predict_traj = len(indices_for_predict)

        # expand senders and receivers (since senders and receivers are the same for all trajectories and frames)
        expanded_senders = np.repeat(np.expand_dims(sender_flags, axis=0), n_of_predict_traj, axis=0)
        expanded_receivers = np.repeat(np.expand_dims(receiver_flags, axis=0), n_of_predict_traj, axis=0)
        expanded_control_flag = np.repeat(np.expand_dims(np.repeat(np.expand_dims(self.control_flags, axis=1), self.n_of_object_attr, axis=1), axis=0), n_of_predict_traj, axis=0) # expand control_flag (21,) -> (traj_num, 21, obj_attr,)

        prediction_objects = objects[indices_for_predict,0,:,:]
        controlled_predictions = objects[indices_for_predict,0,:,:] * expanded_control_flag # for the beginning 
        predicted_objects = np.zeros((n_of_predict_traj, self.n_of_frames-1, self.n_of_points, self.n_of_object_attr)) 
        prediction_errors = []

        for f in range(self.n_of_frames-1):
            if np.max(objects_flag[indices_for_predict,f+1,:,:]) == 0: # break if any one of the trajectories' frame is finished
                break
            
            print('frame_num: {}'.format(f))

            # prepare objects arrays to expand them to have next positions for non-given objects
            propagation = np.zeros((self.n_of_traj, self.n_of_points, n_of_features))
            controlled_target = target[indices_for_predict,f,:,:] * expanded_control_flag # zero for given objects, next positions for non-given objects
            # predicting_objects = np.concatenate((prediction_objects, controlled_target - controlled_predictions), axis=2)
            if chain_effects == False: # the predictions are not used as object attributes in the next steps
                predicting_objects = np.concatenate((objects[indices_for_predict,f,:,:], controlled_target - controlled_predictions), axis=2)
            else:
                predicting_objects = np.concatenate((prediction_objects, controlled_target - controlled_predictions), axis=2)

            predictions = self.gnn_model.predict({'objects': predicting_objects,
                                                  'relations': relations[indices_for_predict,f,:,:],
                                                  'sender_flags': expanded_senders,
                                                  'receiver_flags': expanded_receivers,
                                                  'control_flags': expanded_control_flag,
                                                  'propagation': propagation[indices_for_predict,:,:]})


            # change the batch that will be given for the next prediction by using control flag
            controlled_predictions = predictions * expanded_control_flag # have the joints that are not given, put zeros for the joints that are given   
            controlled_given_objects = target[indices_for_predict,f,:,:] * -(expanded_control_flag-1) # have the joints that are given, put zeros for the joints that are not given
            prediction_objects = controlled_predictions + controlled_given_objects # have them summed elementwise

            # put the prediction in this frame to the predicted_objects
            predicted_objects[:,f,:,:] = prediction_objects[:,:,:] * self.scaling_constant

            # calculate the error in the current frame
            current_error = np.sum(np.absolute(predicted_objects[:,f,:,:] - target[indices_for_predict,f,:,:]*self.scaling_constant))
            prediction_errors.append(current_error)
            print('error in frame {} is: {}'.format(f, current_error))

        # plot the error
        if show_error:
            plt.figure()
            plt.plot(range(len(prediction_errors)), prediction_errors)
            plt.title('Prediction Errors')
            plt.xlabel('Frame')
            plt.ylabel('Error')
            plt.show()

        return predicted_objects, prediction_errors

    # saves gnn_model weights
    def save_gnn(self):
        # serialize weights to h5py file
        weights_file = h5py.File(self.data_path+'gnn_model_weights.h5','w')
        weight = self.gnn_model.get_weights()
        for i in range(len(weight)):
            weights_file.create_dataset('weight'+str(i),data=weight[i])
        weights_file.close()
    
    # loads gnn_model
    def load_gnn(self):

        # get the gnn model
        prop_net = PredictionPropNet()
        self.gnn_model = prop_net.get_model(n_of_frames=self.n_of_frames, n_of_points=self.n_of_points, n_of_relations=self.n_of_relations,
                                            n_of_object_attr=self.n_of_training_attr, n_of_relation_attr=self.n_of_relation_attr, n_of_desired_attr=self.n_of_object_attr) 

        # load the weights
        weights_file = h5py.File(self.data_path+'gnn_model_weights.h5','r')
        weight = []
        for i in range(len(weights_file.keys())):
            weight.append(weights_file['weight'+str(i)][:])
        self.gnn_model.set_weights(weight)
        weights_file.close()

    # if frame_range is None then it shows the whole trajectory otherwise it shows the given frame_range
    # points: positions of each point
    # rows, columns: number of trajectory desired
    # wanted_indices: used with predictions, indices of trajectories that the predictions are created from
    # show_predictions, predictions: boolean and array to show predictions
    # show_relations: boolean to indicate to show/don't show the relations
    # frame_range: frame range to show 
    def visualize(self, points, rows, columns, predicted_objects=None, wanted_indices=None, show_predictions=False, show_relations=False, frame_range=None):

         # get the trajectory number to be taken randomly and take random indices in that range
        n_of_traj = points.shape[0]
        n_of_frames = points.shape[1]
        rand_traj_indices = random.sample(range(n_of_traj), rows*columns)
        if not wanted_indices == None:
            traj_indices = wanted_indices
        else:
            traj_indices = rand_traj_indices
        n_of_traj = len(traj_indices)
        

        # initialize axes and lines
        axes = []
        lines = []
        rel_lines = []
        pre_lines = []
        fig = plt.figure(figsize=(30,30))
        for i in range(1,n_of_traj+1):
            axes.append(fig.add_subplot(rows, columns, i))
        for i in range(n_of_traj):
            # create point lines
            ln, = axes[i].plot([],[], 'ro')
            lines.append(ln)
            
            # create prediction lines
            ln, = axes[i].plot([],[], 'go')
            pre_lines.append(ln)
            
            # create relation lines
            rel_lines.append([])
            if show_relations:
                for _ in range(self.n_of_relations):
                    ln, = axes[i].plot([],[], 'b-')
                    rel_lines[i].append(ln)

        # find minimum and maximum points in moves with rand_traj_indices - this is done to set the limits in axes
        minimums = np.zeros((n_of_traj, 2))
        maximums = np.zeros((n_of_traj, 2))
        for i,t in enumerate(traj_indices):
            minimums[i] = np.nanmin(points[t,:,:,:], axis=(0,1))
            maximums[i] = np.nanmax(points[t,:,:,:], axis=(0,1))

        # initialize the axises
        for i,ax in enumerate(axes):
            ax.set_xlim(minimums[i,0]-700, maximums[i,0]+700)
            ax.set_ylim(minimums[i,1]-200, maximums[i,1]+200)
            ax.set_xlabel('trajectory: {}'.format(traj_indices[i]))
            ax.grid()

        # create animations with predictions
        if show_predictions:

            for i,t in enumerate(traj_indices):
                # find the last frame that is not nan in points, and then find random frame in range(last_frame-frame_frange) as the beginning frame
                last_frame = n_of_frames
                for f in range(n_of_frames):
                    if np.isnan(points[t,f,0,0]) or np.isnan(points[t,f,0,1]):
                        last_frame = f
                        break

                points_and_predictions = np.concatenate((points[t,:last_frame-1,:,:], predicted_objects[i,:last_frame,:,:]), axis=2)
                print('points_and_predictions.shape: {}'.format(points_and_predictions.shape))

                # show the whole trajectory if frame_range is None
                if frame_range == None:
                    FuncAnimation(fig, self.update_visualization,
                                fargs=(i, lines, rel_lines, pre_lines, show_relations, show_predictions),
                                frames=points_and_predictions,
                                interval=15, blit=True, cache_frame_data=False)

        #  create the animations without predictions
        else:
            
            for i,t in enumerate(traj_indices):
                # find the last frame that is not nan in points, and then find random frame in range(last_frame-frame_frange) as the beginning frame
                last_frame = n_of_frames
                for f in range(n_of_frames):
                    if np.isnan(points[t,f,0,0]) or np.isnan(points[t,f,0,1]):
                        last_frame = f
                        break
                
                # show the whole trajectory if frame_range is None
                if frame_range == None:
                    FuncAnimation(fig, self.update_visualization,
                                fargs=(i, lines, rel_lines, pre_lines, show_relations, show_predictions),
                                frames=points[t,:last_frame,:,:],
                                interval=15, blit=True, cache_frame_data=False)

                # otherwise find a random range to show
                else: 
                    if last_frame <= frame_range:
                        start_frame = 0
                    else:
                        start_frame = random.choice(range(last_frame-frame_range))
                    
                    FuncAnimation(fig, self.update_visualization,
                                fargs=(i, lines, rel_lines, pre_lines, show_relations, show_predictions),
                                frames=points[t,start_frame:start_frame+frame_range,:,:],
                                interval=15, blit=True, cache_frame_data=False)

        

        # draw the plot
        plt.show()

    # i: index of the axes
    # frame: data in that trajectory at given frame - frame.shape = ((n_objects, object_dim))
    def update_visualization(self, frame, i, lines, rel_lines, pre_lines, show_relations, show_predictions):

        # set the data for prediction line
        if show_predictions:            
            pre_lines[i].set_data(frame[self.actual_control_indices,2], frame[self.actual_control_indices,3])

        # set the data for actual points
        lines[i].set_data(frame[:,0], frame[:,1])

        # set the data for relation line
        if show_relations:
            relation_index = 0
            for sender_index in range(len(self.actual_hardcoded_relations)):
                for receiver_index in self.actual_hardcoded_relations[sender_index]:
                    x = [frame[sender_index,0], frame[receiver_index,0]]
                    y = [frame[sender_index,1], frame[receiver_index,1]]
                    rel_lines[i][relation_index].set_data(x,y)
                    relation_index += 1

        return lines[i], pre_lines[i],

if __name__ == "__main__":
    hmm_predictor = HumanMotionPredictor(
        max_num_moves=3966,
        dir_path='/home/irmak/Workspaces/lectures_ws/data_git/human_motion_modelling_data/2017-06-22/',
        data_path='/home/irmak/Workspaces/lectures_ws/lectures_git/cmpe_492/human_motion_modelling/data/prediction/',
        max_file_digit=5,
        n_of_points=53,
        n_of_frames=1500)
    
    # hmm_predictor.preprocess_data()
    
    data_path='/home/irmak/Workspaces/lectures_ws/lectures_git/cmpe_492/human_motion_modelling/data/prediction/'
    objects = np.load(open(data_path+'objects.npy', 'rb'))
    n_of_traj = objects.shape[0]
    # hmm_predictor.visualize(objects[:,:,:,[0,2]], 2, 3, show_relations=True)

    # wanted_indices = [99]
    # wanted_indices = [1356, 1145, 389, 99]
    wanted_indices = [1145, 389]
    # wanted_indices = [389, 1451]
    # wanted_indices = list(np.random.choice(np.arange(n_of_traj), 6))
    # print(wanted_indices)
    col = 2
    row = math.ceil(len(wanted_indices)/col)
    # hmm_predictor.visualize(objects[:,:,:,[0,2]], row, col, show_relations=True, wanted_indices=wanted_indices)
    # wanted_objects = objects[wanted_indices,:,:,:]
    # np.save(open(data_path+'wanted_objects.npy', 'wb'), wanted_objects)

    # predicted_objects, training_errors = hmm_predictor.train_gnn(chain_effects=False, indices_for_train=wanted_indices, show_error=False)
    # predicted_objects, prediction_errors = hmm_predictor.make_predictions(chain_effects=False, indices_for_predict=wanted_indices, show_error=True)
    # np.save(open(data_path+'predicted_objects_head_hands_feet_10_one_step.npy', 'wb'), predicted_objects)
    predicted_objects = np.load(open(data_path+'predicted_objects_right_hand_10.npy', 'rb'))
    hmm_predictor.visualize(objects[:,:,:,[1,2]], row,col, predicted_objects=predicted_objects[:,:,:,[1,2]], wanted_indices=wanted_indices, show_relations=True, show_predictions=True)
    


    
    
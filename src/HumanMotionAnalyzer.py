import c3d
import json
import h5py
import math
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import nltk
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

# this class is for mocking gnn_model so that prediction system can be tested before training gnn_model
class MockGnnModel:
    def __init__(self, num_of_class=10):
        self.n_of_class = num_of_class
    
    # returns an array of zeros with one random index 1
    # dict will not be used :)
    def predict(self, silly_dict={}):
        self.n_of_traj = silly_dict['objects'].shape[0]
        random_arr = np.zeros((self.n_of_traj, self.n_of_class))
        for t in range(self.n_of_traj):
            rand_index = random.choice(range(self.n_of_class))
            random_arr[t,rand_index] = 1
        return random_arr

# This class makes a classification for the classes
class HumanMotionAnalyzer:

    # dir_path: path of the directory with all the c3d and json files 
    # max_num_moves: maximum number of movements
    def __init__(self, max_num_moves, dir_path='', max_file_digit=5, max_frame_number=150, max_point_number=55, gnn_model=None):
        self.max_num_moves = max_num_moves
        self.max_file_digit = max_file_digit
        self.dir_path = dir_path
        self.max_frame_number = max_frame_number # frames that are larger than this number is ignored
        self.max_point_number = max_point_number 
        self.max_relation_number = 2 * self.max_point_number # closest 2 points are received as the relation

        self.classes_per_movement = [] # array of classified annotations for each movement available
        self.num_classes_per_movement = []
        self.most_used_classes = [] # array of top 100 classifications 
        if os.path.exists('train_indices.npy') and os.path.exists('test_indices.npy'):
            self.train_indices = np.load(open('train_indices.npy','rb'))
            self.test_indices = np.load(open('test_indices.npy', 'rb'))

        self.gnn_model = gnn_model
        self.object_dim = 2
        self.num_of_object_attr = 2 # distance of each object to the mean of all points
        self.num_of_relation_attr = 2 # represents the distance between objects
        self.relation_threshold = 100
        self.n_of_class = 10
        self.relation_batch_size = 500

        # download natural language toolkit packages
        # nltk.download('universal_tagset')
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')

    # iterates over annotations in the dir_path, and returns a list of classes for each movement
    def classify_movements(self):
        
        for move_num in range(0, self.max_num_moves+1):
            
            str_move_num = str(move_num)
            while len(str_move_num) < self.max_file_digit:
                str_move_num = '0' + str_move_num
            
            file_path = self.dir_path + str_move_num + '_annotations.json'                
            movement_class = '' # string to tag the movement
            movement_num = 0
            
            if os.path.exists(file_path):
                with open(file_path) as json_file:
                    current_annotations = json.load(json_file)
                    
                    st = LancasterStemmer()

                    annotations_sentence = ''
                    for curr_anno in current_annotations:
                        for curr_word in curr_anno.split(' '):
                            annotations_sentence += st.stem(curr_word) + ' '

                    # hardcode the possible 10 classes
                    if 'walk' in annotations_sentence:
                        movement_class = 'walk'
                        movement_num = 0
                    elif 'perform' in annotations_sentence:
                        movement_class = 'perform'
                        movement_num = 1
                    elif 'run' in annotations_sentence or 'jog' in annotations_sentence:
                        movement_class = 'run'
                        movement_num = 2
                    elif 'turn' in annotations_sentence:
                        movement_class = 'turn'
                        movement_num = 3
                    elif 'play' in annotations_sentence:
                        movement_class = 'play'
                        movement_num = 4
                    elif 'going' in annotations_sentence or 'goe' in annotations_sentence:
                        movement_class = 'go'
                        movement_num = 5
                    elif 'wav' in annotations_sentence:
                        movement_class = 'wave'
                        movement_num = 6
                    elif 'mak' in annotations_sentence:
                        movement_class = 'make'
                        movement_num = 7
                    elif 'tak' in annotations_sentence:
                        movement_class = 'take'
                        movement_num = 8
                    elif 'jump' in annotations_sentence:
                        movement_class = 'jump'
                        movement_num = 9

                current_class_exists = False
                for c in self.most_used_classes:
                    if movement_class == c[0]:
                        c[1] += 1
                        current_class_exists = True
                if not current_class_exists:
                    self.most_used_classes.append([movement_class, 1])
                
            self.classes_per_movement.append(movement_class)
            self.num_classes_per_movement.append(movement_num)

        self.most_used_classes.sort(key=operator.itemgetter(1), reverse=True)

        return self.classes_per_movement, self.num_classes_per_movement

    # general visualization method
    # if there are no points saved yet it takes them from the readers if there are points saved then it takes them from saved numpy arrays
    # if there are predictions made then it writes the predictions and the actual classes with them, if there are predictions then it automatically means that it gets the data from test_points
    # it takes a frame range and for each image it finds a random start frame and shows frames starting from that one and until start+range frame, if frame range is None then it shows all of them
    # it can show relations or not accordingly
    def visualize(self, rows, columns, show_train_data=False, data_is_saved=False, show_predictions=False, frame_range=None, show_relations=False):
        # create an array named points with positions of all the points
        if data_is_saved: # saved data is used mostly for showing simple samples with max_frame_number frames, showing predictions and relations
            if show_train_data:
                points = np.load(open('points.npy', 'rb'))
                if show_relations:
                    relations = np.load(open('relations.npy','rb'))
                classes = [self.classes_per_movement[t] for t in self.train_indices]
            else: # show test data
                points = np.load(open('test_points.npy','rb'))
                if show_relations: # relations are shown only if data is already saved, otherwise it is cumbersome to show it
                    relations = np.load(open('test_relations.npy','rb'))
                classes = [self.classes_per_movement[t] for t in self.test_indices]
            
        else: # NOTE: if data is not saved, more frames are used so that random frames can be showed, that is why n_of_frame is 1500
            non_empty_indices = [i for i,c in enumerate(self.classes_per_movement) if c is not '']
            n_of_traj = len(non_empty_indices)
            n_of_frame = 1500 # to be able to show random frames - frame_range is assumed to be not larger then 1500
            n_objects = self.max_point_number
            object_dim = self.object_dim

            # create readers to retrieve the data
            readers = []
            reader_files = []
            for move_num in non_empty_indices:
                # create file name
                str_move_num = str(move_num)
                while len(str_move_num) < self.max_file_digit:
                    str_move_num = '0' + str_move_num
                file_path = self.dir_path + str_move_num + '_raw.c3d'
                reader_file = open(file_path, 'rb')
                reader_files.append(reader_file)
                readers.append(c3d.Reader(reader_file))

            print('n_of_traj before: {}'.format(n_of_traj))
            # check for files that have larger points/frames than maximum num of them and remove them 
            copy_readers = [r for r in readers]
            for i in range(n_of_traj):
                for f, points, _ in copy_readers[i].read_frames():
                    if points.shape[0] > self.max_point_number:
                        readers.remove(copy_readers[i])
                        n_of_traj -= 1
                    break
            print('n_of_traj after: {}'.format(n_of_traj))

            # put all the data into points array
            points = np.zeros((n_of_traj, n_of_frame, n_objects, object_dim))

            # fill points and ignore frames larger 
            for t in range(n_of_traj):

                # ignore larger frames in points 
                last_frame = readers[t].last_frame()
                points[t,last_frame:,:,:] = np.nan
                first_frame = 0

                for f,data_pts,_ in readers[t].read_frames():

                    if first_frame == 0: # some of the data start from larger frames
                        first_frame = f

                    if f-first_frame >= self.max_frame_number:
                        break
                    
                    # ignore larger objects in points 
                    last_obj_num = data_pts.shape[0]
                    points[t,f-first_frame,last_obj_num:,:] = np.nan
                    
                    # put the data into points
                    points[t,f-first_frame,:last_obj_num,:] = data_pts[:,1:3]

            # close the reader_files
            for r in reader_files:
                r.close()     

            classes = [c for c in self.classes_per_movement]     

        print('points.shape: {}, classes.shape: {}'.format(points.shape, len(classes)))

        # get the trajectory number to be taken randomly and take random indices in that range
        n_of_traj = points.shape[0]
        n_of_frame = points.shape[1]
        n_objects = points.shape[2]
        n_of_rand_traj = rows * columns
        rand_traj_indices = random.sample(range(n_of_traj), n_of_rand_traj)

        # initialize axes and lines
        axes = []
        lines = []
        rel_lines = []
        fig = plt.figure(figsize=(30,30))
        for i in range(1,n_of_rand_traj+1):
            axes.append(fig.add_subplot(rows, columns, i))
        for i in range(n_of_rand_traj):
            ln, = axes[i].plot([],[], 'ro')
            lines.append(ln)
            if show_relations:
                ln, = axes[i].plot([],[], 'bo-')
                rel_lines.append(ln)

        # # find minimum and maximum points in moves with rand_traj_indices - this is done to set the limits in axes
        minimums = np.zeros((n_of_rand_traj, 2))
        maximums = np.zeros((n_of_rand_traj, 2))
        for i,t in enumerate(rand_traj_indices):
            minimums[i] = np.nanmin(points[t,:,:,:], axis=(0,1))
            maximums[i] = np.nanmax(points[t,:,:,:], axis=(0,1))

        # initialize the axises
        for i,ax in enumerate(axes):
            ax.set_xlim(minimums[i,0]-700, maximums[i,0]+700)
            ax.set_ylim(minimums[i,1]-200, maximums[i,1]+200)
            # show predicted class under the actual class
            if show_predictions:
                ax.set_xlabel('actual class: ' + classes[rand_traj_indices[i]] + '\n' + 'predicted class: ' + self.flip_class(self.predicted_classes[rand_traj_indices[i]], word_to_arr=False))
            else:
                ax.set_xlabel(classes[rand_traj_indices[i]])
            ax.grid()

        # create the animations
        for i,t in enumerate(rand_traj_indices):
            # find the last frame that is not nan in points, and then find random frame in range(last_frame-frame_frange) as the beginning frame
            last_frame = n_of_frame
            for f in range(n_of_frame):
                if np.isnan(points[t,f,0,0]) or np.isnan(points[t,f,0,1]):
                    last_frame = f
                    break

            # find the last object that is not nan 
            if frame_range == None:
                FuncAnimation(fig, self.update_visualization,
                            fargs=(i, lines, rel_lines, show_relations),
                            frames=points[t,:last_frame,:,:],
                            interval=20, blit=True, cache_frame_data=False)
            else: 
                # find first_frame to start in points
                if last_frame <= frame_range:
                    start_frame = 0
                else:
                    start_frame = random.choice(range(last_frame-frame_range))
                print('start_frame: {}, end_frame: {}, last_frame: {}'.format(start_frame, start_frame+frame_range, last_frame))
                FuncAnimation(fig, self.update_visualization,
                            fargs=(i, lines, rel_lines, show_relations),
                            frames=points[t,start_frame:start_frame+frame_range,:,:],
                            interval=10, blit=True, cache_frame_data=False)  

        # draw the plot
        plt.show()
    
    # i: index of the axes
    # frame: data in that trajectory at given frame - frame.shape = ((n_objects, object_dim))
    def update_visualization(self, frame, i, lines, rel_lines, show_relations):
        # check for objects with nan data
        n_objects = frame.shape[0]
        last_object_num = n_objects
        for o in range(n_objects):
            if np.isnan(frame[o,0]) or np.isnan(frame[o,1]):
                last_object_num = o
                break
        
        if show_relations:
            # TODO: add showing relations here
            x = 5
        
        lines[i].set_data(frame[:last_object_num,0], frame[:last_object_num,1])
        return lines[i],

    # visualizes random rows*columns number of moves and their classes at classes_per_movement
    def visualize_random_moves(self, rows, columns):
        non_empty_indices = [i for i,c in enumerate(self.classes_per_movement) if c is not '']
        num_of_files = rows*columns
        
        print('non_empty_indices[0:100]: {}, num_of_files: {}'.format(non_empty_indices[0:100], num_of_files))
        rand_non_empty_indices = random.sample(non_empty_indices, num_of_files) # get rows*columns times of random indices to draw
        print('rand_non_empty_indices: {}'.format(rand_non_empty_indices))
        tags_of_rand_indices = [self.classes_per_movement[i] for i in rand_non_empty_indices]
        print('tags_of_rand_indices: {}'.format(tags_of_rand_indices))

        # create figure and axes
        axes = []
        data = [] # data = [ [[x1data],[y1data]], [[x2data],[y2data]] ...]
        lines = []
        fig = plt.figure(figsize=(30,30))
        for i in range(1,num_of_files+1):
            axes.append(fig.add_subplot(rows, columns, i))
        for i in range(num_of_files):
            data.append([[],[]])
            ln, = axes[i].plot([],[], 'ro')
            lines.append(ln)

        # create readers to retrieve the data
        readers = []
        reader_files = []
        for move_num in rand_non_empty_indices:
            # create file name
            str_move_num = str(move_num)
            while len(str_move_num) < self.max_file_digit:
                str_move_num = '0' + str_move_num
            file_path = self.dir_path + str_move_num + '_raw.c3d'
            reader_file = open(file_path, 'rb')
            reader_files.append(reader_file)
            readers.append(c3d.Reader(reader_file))

        xmins = [sys.maxsize for _ in range(num_of_files)]
        xmaxes = [-sys.maxsize for _ in range(num_of_files)]
        ymins = [sys.maxsize for _ in range(num_of_files)]
        ymaxes = [-sys.maxsize for _ in range(num_of_files)]
        for i in range(num_of_files):
            for _, points, _ in readers[i].read_frames():
                xmins[i] = min(xmins[i], min(points[:,1]))
                ymins[i] = min(ymins[i], min(points[:,2]))
                xmaxes[i] = max(xmaxes[i], max(points[:,1]))
                ymaxes[i] = max(ymaxes[i], max(points[:,2]))

        # initialize the axises
        for i,ax in enumerate(axes):
            ax.set_xlim(xmins[i]-700, xmaxes[i]+700)
            ax.set_ylim(ymins[i]-200, ymaxes[i]+200)
            ax.set_xlabel(tags_of_rand_indices[i])
            ax.grid()

        # create the animations
        for i in range(num_of_files):
            FuncAnimation(fig, self.update_visualization_data, fargs=(i,data,lines), frames=readers[i].read_frames(), interval=10, blit=True, cache_frame_data=False)

        # draw the plot
        plt.show()

        # close the files
        for r in reader_files:
            r.close()

    # it visualizes n_of_frame number of frame of random moves by reading points.npy file
    def visualize_random_moves_points(self, rows, columns, frame_range):
        num_of_moves = rows * columns
        points = np.load(open('points.npy','rb'))
        
        # create figure and axes
        axes = []
        data = [] # data = [ [[x1data],[y1data]], [[x2data],[y2data]] ...]
        lines = []
        fig = plt.figure(figsize=(30,30))
        for i in range(1,num_of_moves+1):
            axes.append(fig.add_subplot(rows, columns, i))
        for i in range(num_of_moves):
            data.append([[],[]])
            ln, = axes[i].plot([],[], 'ro')
            lines.append(ln)

        # find minimum values
        xmins = [sys.maxsize for _ in range(num_of_files)]
        xmaxes = [-sys.maxsize for _ in range(num_of_files)]
        ymins = [sys.maxsize for _ in range(num_of_files)]
        ymaxes = [-sys.maxsize for _ in range(num_of_files)]
        for i in range(num_of_files):
            for _, points, _ in readers[i].read_frames():
                xmins[i] = min(xmins[i], min(points[:,1]))
                ymins[i] = min(ymins[i], min(points[:,2]))
                xmaxes[i] = max(xmaxes[i], max(points[:,1]))
                ymaxes[i] = max(ymaxes[i], max(points[:,2]))

        # initialize the axises
        for i,ax in enumerate(axes):
            ax.set_xlim(xmins[i]-700, xmaxes[i]+700)
            ax.set_ylim(ymins[i]-200, ymaxes[i]+200)
            ax.set_xlabel(tags_of_rand_indices[i])
            ax.grid()

    # i: index of the axes
    # frame: frame received from the reader[i]
    def update_visualization_data(self, frame, i, data, lines):
        points = frame[1]
        data[i][0] = points[:,1]
        data[i][1] = points[:,2]

        lines[i].set_data(data[i][0], data[i][1])
        return lines[i], 
      
    # trains the graph neural network TODO: have a test data, rn model is trained with the whole data
    def train_gnn(self, relations_batch_number=1):

        objects = np.load(open('objects.npy','rb')) # this one loads all of the trajectories
        if relations_batch_number == 1:
            relations = np.load(open('relations.npy','rb'))
            sender_flags = np.load(open('senders.npy','rb'))
            receiver_flags = np.load(open('receivers.npy', 'rb'))
        else:
            relations = np.load(open('relations' + str(relations_batch_number) + '.npy','rb'))
            sender_flags = np.load(open('senders' + str(relations_batch_number) + '.npy','rb'))
            receiver_flags = np.load(open('receivers' + str(relations_batch_number) + '.npy', 'rb'))

        print('arrays loaded')
        n_of_traj = relations.shape[0] # for the reasons explained above
        n_of_frame = relations.shape[1]
        n_of_relations = relations.shape[2]
        n_objects = objects.shape[2]
        n_of_features = 100

        propagation = np.zeros((n_of_traj, n_of_frame, n_objects, n_of_features))

        # get the class for each movement as the target
        # and enumerate them such that it will be [0,0,0,1,0,0,0,0,0,0] rather than simply being 3
        train_y = [self.num_classes_per_movement[t] for t in self.train_indices]
        enumerated_y = np.zeros((n_of_traj, self.n_of_class))
        num_of_walks = 0
        for t in range(n_of_traj):
            enumerated_y[t, train_y[t]] = 1
            if train_y[t] == 0:
                num_of_walks += 1
        print('num_of_walks: {}'.format(num_of_walks))
        print('enumerated_y: {}'.format(enumerated_y))

        max_distance = 2000 # TODO: check this value out 
        objects = objects / max_distance
        relations = relations / max_distance

        # create or load the graph neural network model
        if relations_batch_number != 1:
            self.load_gnn()
        else:
            prop_net = PropagationNetwork()
            self.gnn_model = prop_net.getModel(n_of_frame=n_of_frame, n_objects=n_objects, n_of_relations=n_of_relations, n_of_class=self.n_of_class, n_of_features=n_of_features)

        print('training model from trajectory: {}, to trajectory: {}'.format((relations_batch_number-1) * self.relation_batch_size, relations_batch_number * self.relation_batch_size))

        # train the model
        self.gnn_model.fit({'objects': objects[(relations_batch_number-1) * self.relation_batch_size : relations_batch_number * self.relation_batch_size,:,:,:],
                            'relations': relations,
                            'sender_flags': sender_flags,
                            'receiver_flags': receiver_flags,
                            'propagation': propagation},
                            {'target': enumerated_y},
                            batch_size=5,
                            epochs=60,
                            validation_split=0.2,
                            shuffle=True,
                            verbose=1)

        # save gnn_model
        self.save_model()

        return self.gnn_model

    # saves gnn_model weights
    def save_model(self):
        # serialize weights to h5py file
        weights_file = h5py.File('gnn_model_weights.h5','w')
        weight = self.gnn_model.get_weights()
        for i in range(len(weight)):
            weights_file.create_dataset('weight'+str(i),data=weight[i])
        weights_file.close()
    
    # loads gnn_model
    def load_gnn(self):

        # create the graph neural network model
        prop_net = PropagationNetwork()
        self.gnn_model = prop_net.getModel(n_of_frame=self.max_frame_number, n_objects=self.max_point_number,
                                           n_of_relations=self.max_relation_number, n_of_class=self.n_of_class, n_of_features=100)

        # load the weights
        weights_file = h5py.File('gnn_model_weights.h5','r')
        weight = []
        for i in range(len(weights_file.keys())):
            weight.append(weights_file['weight'+str(i)][:])
        self.gnn_model.set_weights(weight)
        weights_file.close()

    # this method preprocesses the data, decreases the number of frames
    # fixes the problem of dynamic number of points 
    # and saves numpy arrays representing objects, senders, receivers and relations 
    def preprocess_data(self):

        if (os.path.exists('points.npy')):
            points = np.load(open('points.npy', 'rb'))
            objects_flag = np.load(open('objects_flag.npy', 'rb'))
            
            n_of_traj = points.shape[0]
            n_of_frame = points.shape[1]
            n_objects = points.shape[2]
            object_dim = points.shape[3]
        
            print('points and objects_flag loaded')

        else:

            self.non_empty_indices = [i for i,c in enumerate(self.classes_per_movement) if c is not '']
            self.num_of_moves = len(self.non_empty_indices)
            self.train_indices = random.sample(self.non_empty_indices, int(self.num_of_moves * 70.0 / 100.0)) # 70% of the data for training
            self.test_indices = [i for i in self.non_empty_indices if i not in self.train_indices]

            # create the readers to receive the data
            readers = []
            reader_files = []
            for move_num in self.train_indices:
                # create file name
                str_move_num = str(move_num)
                while len(str_move_num) < self.max_file_digit:
                    str_move_num = '0' + str_move_num
                file_path = self.dir_path + str_move_num + '_raw.c3d'
                reader_file = open(file_path, 'rb')
                reader_files.append(reader_file)
                readers.append(c3d.Reader(reader_file))

            # check for files that have larger points/frames than maximum num of them and remove them 
            print('len(train_indices) before: {}, len(readers) before: {}'.format(len(self.train_indices), len(readers)))
            copy_train_indices = [i for i in self.train_indices]
            copy_readers = [r for r in readers]
            for i in range(len(self.train_indices)):
                for f, points, _ in copy_readers[i].read_frames():
                    if points.shape[0] > self.max_point_number:
                        self.train_indices.remove(copy_train_indices[i])
                        readers.remove(copy_readers[i])
                    break
            print('len(train_indices) before: {}, len(readers) before: {}'.format(len(self.train_indices), len(readers)))

            self.train_indices = np.array(self.train_indices)
            np.save(open('train_indices.npy','wb'), self.train_indices)
            self.test_indices = np.array(self.test_indices)
            np.save(open('test_indices.npy','wb'), self.test_indices)

            # set the number of dimensions for the points array (array with the positions of each object)
            n_of_traj = len(self.train_indices)
            n_of_frame = self.max_frame_number
            n_objects = self.max_point_number
            object_dim = self.object_dim

            print('after removing points/frames: n_of_traj: {}, n_of_frame: {}, n_objects: {},'.format(n_of_traj, n_of_frame, n_objects))

            # put the data into numpy array
            points = np.zeros((n_of_traj, n_of_frame, n_objects, object_dim))
            objects_flag = np.zeros((n_of_traj, n_of_frame, n_objects, 1)) # 1-0 flag, 0 when there are not enough frames or points in the current trajectory
            for t in range(n_of_traj):

                # ignore larger frames in points 
                last_frame = readers[t].last_frame()
                points[t,last_frame:,:,:] = np.nan
                first_frame = 0

                for f,data_pts,_ in readers[t].read_frames():

                    if first_frame == 0: # some of the data start from larger frames
                        first_frame = f

                    if f-first_frame >= self.max_frame_number:
                        break
                    
                    # ignore larger objects in points 
                    last_obj_num = data_pts.shape[0]
                    points[t,f-first_frame,last_obj_num:,:] = np.nan
                    
                    # put the data into points
                    points[t,f-first_frame,:last_obj_num,:] = data_pts[:,1:3]
                    objects_flag[t,f-first_frame,:last_obj_num,:] = 1

            # close the reader_files
            for r in reader_files:
                r.close()

            # save created arrays
            np.save(open('points.npy', 'wb'), points)
            np.save(open('objects_flag.npy', 'wb'), objects_flag)

            print('points and objects_flag created')

        
        if (os.path.exists('objects.npy')):
            objects = np.load(open('objects.npy', 'rb'))
            print('objects loaded')
        
        else:
            # create the objects array
            # objects array will have for each point, the distance of the points to the mean of all points for that frame
            objects = np.zeros((n_of_traj, n_of_frame, n_objects, self.num_of_object_attr))
            print('objects.shape: {}'.format(objects.shape))
            for t in range(n_of_traj):
                for f in range(n_of_frame):
                    x_mean = np.nanmean(points[t,f,:,0])
                    y_mean = np.nanmean(points[t,f,:,1])
                    for o in range(n_objects):
                        if np.isnan(points[t,f,o,0]) or np.isnan(points[t,f,o,1]): # since objects array will be used afterwards nan values are not wanted
                            break
                        objects[t,f,o,0] = points[t,f,o,0] - x_mean
                        objects[t,f,o,1] = points[t,f,o,1] - y_mean
            
            np.save(open('objects.npy', 'wb'), objects)

            print('objects created')

        relation_batch_size = self.relation_batch_size
        if (os.path.exists('relations.npy')):
            # relations = np.load(open('relations.npy', 'rb'))
            # senders = np.load(open('senders.npy', 'rb'))
            # receivers = np.load(open('receivers.npy', 'rb'))
            
            # print('relations and flags loaded')

            n_of_relations = self.max_relation_number

            completed_traj = relation_batch_size
            b = 2
            while completed_traj < n_of_traj: 
                print('****** batch number: {} *******'.format(b))

                end_traj = completed_traj + relation_batch_size
                if end_traj > n_of_traj:
                    end_traj = n_of_traj

                print('end_traj - completed_traj: {}'.format(end_traj - completed_traj))

                # save the second parts of relations and flags
                relations = np.zeros((end_traj - completed_traj, n_of_frame, n_of_relations, self.num_of_relation_attr))
                senders = np.zeros((end_traj - completed_traj, n_of_frame, n_objects, n_of_relations)) # flag to indicate which object belongs to which relations TODO: maybe you can change the dimensions here
                receivers = np.zeros((end_traj - completed_traj, n_of_frame, n_objects, n_of_relations))
            
                for t in range(completed_traj, end_traj):
                    for f in range(n_of_frame):
                        if objects_flag[t,f,0,0] == 0:
                            break
                        for m in range(n_objects):
                            if objects_flag[t,f,m,0] == 0:
                                break
                            shortest_dists = [[sys.maxsize,0,0], [sys.maxsize,0,0]] # closest 2 points are put as relations [euclidean_dist, x_diff, y_diff]
                            closest_points = [0,0]
                            for j in range(n_objects):
                                if objects_flag[t,f,m,0] == 0:
                                    break
                                if m != j:
                                    x_diff = points[t,f,m,0] - points[t,f,j,0]
                                    y_diff = points[t,f,m,1] - points[t,f,j,1]
                                    dist = math.sqrt(x_diff**2 + y_diff**2)
                                    if dist < shortest_dists[0][0]:
                                        # move the second closest one to the second
                                        shortest_dists[1][0] = shortest_dists[0][0]
                                        shortest_dists[1][1] = shortest_dists[0][1]
                                        shortest_dists[1][2] = shortest_dists[0][2]
                                        closest_points[1] = closest_points[0]
                                        # have the closest one on top
                                        shortest_dists[0][0] = dist
                                        shortest_dists[0][1] = x_diff
                                        shortest_dists[0][2] = y_diff
                                        closest_points[0] = j
                                    elif dist < shortest_dists[1][0]:
                                        # record the second closes one
                                        shortest_dists[1][0] = dist
                                        shortest_dists[1][1] = x_diff
                                        shortest_dists[1][2] = y_diff
                                        closest_points[1] = j

                            curr_relation_nums = [m*2, m*2+1]
                            
                            for i in range(2):
                                relations[t-completed_traj,f,curr_relation_nums[i],0] = shortest_dists[i][1]
                                relations[t-completed_traj,f,curr_relation_nums[i],1] = shortest_dists[i][2]
                                senders[t-completed_traj,f,m,curr_relation_nums[i]] = 1
                                receivers[t-completed_traj,f,closest_points[i],curr_relation_nums[i]] = 1
                                
                    print('i: {}, done with the trajectory: {}'.format(b, t))

                # save the created arrays
                np.save(open('relations' + str(b) + '.npy', 'wb'), relations)
                np.save(open('senders' + str(b) + '.npy', 'wb'), senders)
                np.save(open('receivers' + str(b) + '.npy', 'wb'), receivers)
                print('relations' + str(b) + ' and flags' + str(b) + ' created')

                # update the parameters
                completed_traj = end_traj
                b += 1
            
            # relations2 = np.zeros((n_of_traj-relation_batch_size, n_of_frame, n_of_relations, self.num_of_relation_attr))
            # senders2 = np.zeros((n_of_traj-relation_batch_size, n_of_frame, n_objects, n_of_relations)) # flag to indicate which object belongs to which relations TODO: maybe you can change the dimensions here
            # receivers2 = np.zeros((n_of_traj-relation_batch_size, n_of_frame, n_objects, n_of_relations))
            # for t in range(relation_batch_size,n_of_traj):
            #     for f in range(n_of_frame):
            #         if objects_flag[t,f,0,0] == 0:
            #             break
            #         for m in range(n_objects):
            #             if objects_flag[t,f,m,0] == 0:
            #                 break
            #             shortest_dists = [[sys.maxsize,0,0], [sys.maxsize,0,0]] # closest 2 points are put as relations [euclidean_dist, x_diff, y_diff]
            #             closest_points = [0,0]
            #             for j in range(n_objects):
            #                 if objects_flag[t,f,m,0] == 0:
            #                     break
            #                 if m != j:
            #                     x_diff = points[t,f,m,0] - points[t,f,j,0]
            #                     y_diff = points[t,f,m,1] - points[t,f,j,1]
            #                     dist = math.sqrt(x_diff**2 + y_diff**2)
            #                     if dist < shortest_dists[0][0]:
            #                         # move the second closest one to the second
            #                         shortest_dists[1][0] = shortest_dists[0][0]
            #                         shortest_dists[1][1] = shortest_dists[0][1]
            #                         shortest_dists[1][2] = shortest_dists[0][2]
            #                         closest_points[1] = closest_points[0]
            #                         # have the closest one on top
            #                         shortest_dists[0][0] = dist
            #                         shortest_dists[0][1] = x_diff
            #                         shortest_dists[0][2] = y_diff
            #                         closest_points[0] = j
            #                     elif dist < shortest_dists[1][0]:
            #                         # record the second closes one
            #                         shortest_dists[1][0] = dist
            #                         shortest_dists[1][1] = x_diff
            #                         shortest_dists[1][2] = y_diff
            #                         closest_points[1] = j

            #             curr_relation_nums = [m*2, m*2+1]
                        
            #             for i in range(2):
            #                 relations2[t-relation_batch_size,f,curr_relation_nums[i],0] = shortest_dists[i][1]
            #                 relations2[t-relation_batch_size,f,curr_relation_nums[i],1] = shortest_dists[i][2]
            #                 senders2[t-relation_batch_size,f,m,curr_relation_nums[i]] = 1
            #                 receivers2[t-relation_batch_size,f,closest_points[i],curr_relation_nums[i]] = 1
                            
            #     print('done with the trajectory: {}'.format(t))

            # # save the created arrays
            # np.save(open('relations2.npy', 'wb'), relations2)
            # np.save(open('senders2.npy', 'wb'), senders2)
            # np.save(open('receivers2.npy', 'wb'), receivers2)
            # print('relations2 and flags2 created')

        else:

            # create the relations array
            # relations array will have distance bw points for each relation

            n_of_relations = self.max_relation_number

            relations = np.zeros((relation_batch_size, n_of_frame, n_of_relations, self.num_of_relation_attr))
            senders = np.zeros((relation_batch_size, n_of_frame, n_objects, n_of_relations)) # flag to indicate which object belongs to which relations TODO: maybe you can change the dimensions here
            receivers = np.zeros((relation_batch_size, n_of_frame, n_objects, n_of_relations))

            # calculate relations - each point has relationship with two closest points
            for t in range(relation_batch_size):
                for f in range(n_of_frame):
                    if objects_flag[t,f,0,0] == 0:
                        break
                    for m in range(n_objects):
                        if objects_flag[t,f,m,0] == 0:
                            break
                        shortest_dists = [[sys.maxsize,0,0], [sys.maxsize,0,0]] # closest 2 points are put as relations [euclidean_dist, x_diff, y_diff]
                        closest_points = [0,0]
                        for j in range(n_objects):
                            if objects_flag[t,f,m,0] == 0:
                                break
                            if m != j:
                                x_diff = points[t,f,m,0] - points[t,f,j,0]
                                y_diff = points[t,f,m,1] - points[t,f,j,1]
                                dist = math.sqrt(x_diff**2 + y_diff**2)
                                if dist < shortest_dists[0][0]:
                                    # move the second closest one to the second
                                    shortest_dists[1][0] = shortest_dists[0][0]
                                    shortest_dists[1][1] = shortest_dists[0][1]
                                    shortest_dists[1][2] = shortest_dists[0][2]
                                    closest_points[1] = closest_points[0]
                                    # have the closest one on top
                                    shortest_dists[0][0] = dist
                                    shortest_dists[0][1] = x_diff
                                    shortest_dists[0][2] = y_diff
                                    closest_points[0] = j
                                elif dist < shortest_dists[1][0]:
                                    # record the second closes one
                                    shortest_dists[1][0] = dist
                                    shortest_dists[1][1] = x_diff
                                    shortest_dists[1][2] = y_diff
                                    closest_points[1] = j

                        curr_relation_nums = [m*2, m*2+1]
                        for i in range(2):
                            relations[t,f,curr_relation_nums[i],0] = shortest_dists[i][1]
                            relations[t,f,curr_relation_nums[i],1] = shortest_dists[i][2]
                            senders[t,f,m,curr_relation_nums[i]] = 1
                            receivers[t,f,closest_points[i],curr_relation_nums[i]] = 1
                            
                print('done with the trajectory: {}'.format(t))

            # save the created arrays
            np.save(open('relations.npy', 'wb'), relations)
            np.save(open('senders.npy', 'wb'), senders)
            np.save(open('receivers.npy', 'wb'), receivers)
            print('relations and flags created')

    # this method visualizes random relations and the objects by loading the arrays from npy files
    # it gets the mean of the points and visualizes dots by using objects
    # it visualizes relations by adding up to the points
    def visualize_relations(self):
        trajectories = range(20) # TODO: after relations are saved for every trajectory make this range(n_of_traj)
        num_of_traj = 10
        frame = 0

        relations_file = open('relations.npy', 'rb')
        relations = np.load(relations_file)

        objects_file = open('objects.npy', 'rb')
        objects = np.load(objects_file)

        points_file = open('points.npy', 'rb')
        points = np.load(points_file)

        # draw dots
        plt.figure(figsize=(30,30))
        random_trajectories = random.sample(trajectories, num_of_traj)
        for i,t in enumerate(random_trajectories): # this method will draw 5 trajectories side by side
            # get the actual places of dots
            x_mean = np.nanmean(points[t,frame,:,0])
            y_mean = np.nanmean(points[t,frame,:,1])
            x = x_mean + objects[t,frame,:,0] # x: np array with for each objects position
            y = y_mean + objects[t,frame,:,1]
            plt.subplot(1, num_of_traj, i+1)
            plt.plot(x,y,'ro')
            plt.xlabel('trajectory: {}'.format(t))

        # draw relations: find if that distance is in that relation and add to that point and draw a line there
        n_objects = objects.shape[2]
        for i,t in enumerate(random_trajectories):
            for r in range(relations.shape[2]):
                if relations[t,frame,r,0] == 0 and relations[t,frame,r,1] == 0: # could have also checked relations_flag
                    continue 
                for m in range(n_objects):
                    for j in range(n_objects):
                        if m != j:
                            x_diff = points[t,frame,m,0] - points[t,frame,j,0]
                            y_diff = points[t,frame,m,1] - points[t,frame,j,1]
                            if relations[t,frame,r,0] == x_diff and relations[t,frame,r,1] == y_diff:
                                x = [points[t,frame,m,0], points[t,frame,m,0]-x_diff]
                                y = [points[t,frame,m,1], points[t,frame,m,1]-y_diff]
                                plt.subplot(1, num_of_traj, i+1)
                                plt.plot(x,y,'bo-')

                    


        plt.show()

    # make predictions and calculate results in test_indices
    # visualize predicted and random classes too if wanted
    def make_predictions(self, num_of_traj_wanted, visualize=False):

        if os.path.exists('test_predictions.npy'):
            self.predicted_classes = np.load(open('test_predictions.npy','rb'))
            print('predicted_classes loaded')
        
        else:

            # calculate actual values of classes
            self.test_indices = self.test_indices.tolist() # return this np array to a list so that it can be removed when needed
            test_y = [self.num_classes_per_movement[t] for t in self.test_indices]
            enumerated_y = np.zeros((len(test_y), self.n_of_class))
            for t in range(len(test_y)):
                enumerated_y[t, test_y[t]] = 1

            
            n_of_frame = self.max_frame_number
            n_objects = self.max_point_number
            n_of_relations = self.max_relation_number
            object_dim = self.object_dim
            object_attr = self.num_of_object_attr
            relation_attr = self.num_of_relation_attr

            # preprocess test data - create objects, relations, sender and receiver flags for test indices
            # create points and object_flags
            if os.path.exists('test_points.npy') and os.path.exists('test_object_flags.npy'):
                test_points = np.load(open('test_points.npy', 'rb'))
                # test_object_flags = np.load(open('test_object_flags.npy', 'rb'))
                n_of_traj = test_points.shape[0]
                print('points and object_flags loaded')
            else:
                # create the readers to receive the data
                readers = []
                reader_files = []
                for move_num in self.test_indices:
                    # create file name
                    str_move_num = str(move_num)
                    while len(str_move_num) < self.max_file_digit:
                        str_move_num = '0' + str_move_num
                    file_path = self.dir_path + str_move_num + '_raw.c3d'
                    reader_file = open(file_path, 'rb')
                    reader_files.append(reader_file)
                    readers.append(c3d.Reader(reader_file))

                # check for files that have larger points/frames than maximum num of them and remove them 
                print('len(test_indices) before: {}, len(readers) before: {}'.format(len(self.test_indices), len(readers)))
                copy_test_indices = [i for i in self.test_indices]
                copy_readers = [r for r in readers]
                for i in range(len(self.test_indices)):
                    for f, points, _ in copy_readers[i].read_frames():
                        if points.shape[0] > self.max_point_number:
                            self.test_indices.remove(copy_test_indices[i])
                            readers.remove(copy_readers[i])
                        break
                print('len(test_indices) before: {}, len(readers) before: {}'.format(len(self.test_indices), len(readers)))

                self.test_indices = np.array(self.test_indices)
                np.save(open('test_indices.npy','wb'), self.test_indices)

                n_of_traj = len(test_y)

                # put the data into numpy array
                test_points = np.zeros((n_of_traj, n_of_frame, n_objects, object_dim))
                test_objects_flag = np.zeros((n_of_traj, n_of_frame, n_objects, 1)) # 1-0 flag, 0 when there are not enough frames or points in the current trajectory
                for t in range(n_of_traj):

                    # ignore larger frames in points 
                    last_frame = readers[t].last_frame()
                    test_points[t,last_frame:,:,:] = np.nan
                    first_frame = 0

                    for f,data_pts,_ in readers[t].read_frames():

                        if first_frame == 0: # some of the data start from larger frames
                            first_frame = f

                        if f-first_frame >= self.max_frame_number:
                            break
                        
                        # ignore larger objects in points 
                        last_obj_num = data_pts.shape[0]
                        test_points[t,f-first_frame,last_obj_num:,:] = np.nan
                        
                        # put the data into points
                        test_points[t,f-first_frame,:last_obj_num,:] = data_pts[:,1:3]
                        test_objects_flag[t,f-first_frame,:last_obj_num,:] = 1

                # close the reader_files
                for r in reader_files:
                    r.close()

                # save created arrays
                np.save(open('test_points.npy', 'wb'), test_points)
                np.save(open('test_objects_flag.npy', 'wb'), test_objects_flag)

                print('points and objects_flag saved')

            # create objects
            if os.path.exists('test_objects.npy'):
                test_objects = np.load(open('test_objects.npy','rb'))
                print('objects loaded')
            else: 
                test_objects = np.zeros((n_of_traj, n_of_frame, n_objects, object_attr))
                for t in range(n_of_traj):
                    for f in range(n_of_frame):
                        x_mean = np.nanmean(test_points[t,f,:,0])
                        y_mean = np.nanmean(test_points[t,f,:,1])
                        for o in range(n_objects):
                            if np.isnan(test_points[t,f,o,0]) or np.isnan(test_points[t,f,o,1]): # since objects array will be used afterwards nan values are not wanted
                                break
                            test_objects[t,f,o,0] = test_points[t,f,o,0] - x_mean
                            test_objects[t,f,o,1] = test_points[t,f,o,1] - y_mean
                
                np.save(open('test_objects.npy', 'wb'), test_objects)
                print('test_objects saved')

            # create relations and sender/receiver flags
            if os.path.exists('test_relations.npy') and os.path.exists('test_senders.npy') and os.path.exists('test_receivers.npy'):
                test_relations = np.load(open('test_relations.npy','rb'))
                test_senders = np.load(open('test_senders.npy','rb'))
                test_receivers = np.load(open('test_receivers.npy','rb'))
                print('test relations and sender/receiver flags loaded')
            else: 
                test_relations = np.zeros((n_of_traj, n_of_frame, n_of_relations, relation_attr))
                test_senders = np.zeros((n_of_traj, n_of_frame, n_objects, n_of_relations)) # flag to indicate which object belongs to which relations TODO: maybe you can change the dimensions here
                test_receivers = np.zeros((n_of_traj, n_of_frame, n_objects, n_of_relations))
                
                # calculate relations - each point has relationship with two closest points
                for t in range(n_of_traj):
                    for f in range(n_of_frame):
                        if test_objects_flag[t,f,0,0] == 0:
                            break
                        for m in range(n_objects):
                            if test_objects_flag[t,f,m,0] == 0:
                                break
                            shortest_dists = [[sys.maxsize,0,0], [sys.maxsize,0,0]] # closest 2 points are put as relations [euclidean_dist, x_diff, y_diff]
                            closest_points = [0,0]
                            for j in range(n_objects):
                                if test_objects_flag[t,f,m,0] == 0:
                                    break
                                if m != j:
                                    x_diff = test_points[t,f,m,0] - test_points[t,f,j,0]
                                    y_diff = test_points[t,f,m,1] - test_points[t,f,j,1]
                                    dist = math.sqrt(x_diff**2 + y_diff**2)
                                    if dist < shortest_dists[0][0]:
                                        # move the second closest one to the second
                                        shortest_dists[1][0] = shortest_dists[0][0]
                                        shortest_dists[1][1] = shortest_dists[0][1]
                                        shortest_dists[1][2] = shortest_dists[0][2]
                                        closest_points[1] = closest_points[0]
                                        # have the closest one on top
                                        shortest_dists[0][0] = dist
                                        shortest_dists[0][1] = x_diff
                                        shortest_dists[0][2] = y_diff
                                        closest_points[0] = j
                                    elif dist < shortest_dists[1][0]:
                                        # record the second closes one
                                        shortest_dists[1][0] = dist
                                        shortest_dists[1][1] = x_diff
                                        shortest_dists[1][2] = y_diff
                                        closest_points[1] = j

                            curr_relation_nums = [m*2, m*2+1]
                            for i in range(2):
                                test_relations[t,f,curr_relation_nums[i],0] = shortest_dists[i][1]
                                test_relations[t,f,curr_relation_nums[i],1] = shortest_dists[i][2]
                                test_senders[t,f,m,curr_relation_nums[i]] = 1
                                test_receivers[t,f,closest_points[i],curr_relation_nums[i]] = 1
                                
                    print('done with the trajectory: {}'.format(t))
                
                np.save(open('test_relations.npy', 'wb'), test_relations)
                np.save(open('test_senders.npy', 'wb'), test_senders)
                np.save(open('test_receivers.npy', 'wb'), test_receivers)
                print('test relations and sender/receiver flags saved')

            n_of_traj = num_of_traj_wanted

            test_objects = test_objects[:n_of_traj,:,:,:]
            test_relations = test_relations[:n_of_traj,:,:,:]
            test_senders = test_senders[:n_of_traj,:,:,:]
            test_receivers = test_receivers[:n_of_traj,:,:,:]
            propagation = np.zeros((n_of_traj, n_of_frame, n_objects, 100))

            # predict classes
            # with this, predicted classes are probabilities guessed by the model and they are floats btw 0-1
            self.predicted_classes = self.gnn_model.predict({'objects': test_objects, 'relations': test_relations, 'sender_flags': test_senders, 
                                                        'receiver_flags': test_receivers, 'propagation': propagation})

            # save predictions
            np.save(open('test_predictions.npy','wb'), self.predicted_classes)
            print('predicted_classes saved')

        # calculate the accuracy between predicted classes and enumerated_y
        all_predictions = n_of_traj
        correct_predictions = 0
        print('predicted_classes.sample(10): {}'.format(random.sample(10, self.predicted_classes)))
        print('predicted_classes.shape: {}, enumerated_y.shape: {}'.format(self.predicted_classes.shape, enumerated_y.shape))
        for t in range(n_of_traj):
            predicted_correct = True
            for c in range(self.n_of_class):

                # change them into 0s and 1s
                if self.predicted_classes[t,c] >= 0.5:
                    self.predicted_classes[t,c] = 1
                else:
                    self.predicted_classes[t,c] = 0

                if self.predicted_classes[t,c] != enumerated_y[t,c]:
                    predicted_correct = False
                    
            if predicted_correct:
                correct_predictions += 1
        print('correct_predictions / all_predictions: {} / {}, {}'.format(correct_predictions, all_predictions, correct_predictions/all_predictions))

    # gets a word or an array of 0s and 1s as class and flips it in to the other format
    def flip_class(self, word_or_arr, word_to_arr):
        class_dict = {'walk':0, 'perform':1, 'run':2, 'turn':3, 'play':4, 'go':5, 'wave':6, 'make':7, 'take':8, 'jump':9}      
        if word_to_arr:
            arr = np.zeros(self.n_of_class)
            movement_num = class_dict[word_or_arr]
            arr[movement_num] = 1
            return arr
        else: 
            for i,c in enumerate(word_or_arr):
                if c == 1:
                    movement_num = i
            return list(class_dict.keys())[list(class_dict.values()).index(movement_num)]

if __name__ == "__main__":
    # mock_gnn_model = MockGnnModel()
    human_motion_analyzer = HumanMotionAnalyzer(max_num_moves=3966, dir_path='/home/irmak/Workspaces/lectures_ws/data_git/human_motion_modelling_data/2017-06-22/') 
    _, _ = human_motion_analyzer.classify_movements()
    # human_motion_analyzer.visualize(2,5, frame_range=300)
    human_motion_analyzer.load_gnn()
    human_motion_analyzer.make_predictions(678)
    human_motion_analyzer.visualize(2,5, show_train_data=False, show_predictions=True, data_is_saved=True)
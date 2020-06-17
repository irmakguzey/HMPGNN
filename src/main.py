# from HumanMotionAnalyzer import *
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

data_path='/home/irmak/Workspaces/lectures_ws/lectures_git/cmpe_492/human_motion_modelling/data/prediction/'
wanted_objects = np.load(open(data_path+'wanted_objects.npy', 'rb'))
predicted_objects = np.load(open(data_path+'predicted_objects_right_hand_10.npy', 'rb'))
frame_length = 800
joint_size = predicted_objects.shape[2]

# increase the z axis to visualize the system better
# increase = np.linspace((0,0,0), (frame_length, frame_length, frame_length) )

wanted_traj = wanted_objects[1, 1:1+frame_length, :, :]
predicted_traj = predicted_objects[1, :frame_length, :, :]

walk_start_frame = 150
walk_end_frame = 650
x_increase = np.linspace([500 for _ in range(joint_size)], [-300 for _ in range(joint_size)], walk_end_frame - walk_start_frame)

wanted_traj[:walk_start_frame,:,0] = wanted_traj[:walk_start_frame, :,0] + 500
wanted_traj[walk_start_frame:walk_end_frame, :, 0] = wanted_traj[walk_start_frame:walk_end_frame, :, 0] + x_increase
wanted_traj[walk_end_frame:,:,0] = wanted_traj[walk_end_frame:, :,0] - 300

predicted_traj[:walk_start_frame,:,0] = predicted_traj[:walk_start_frame, :,0] + 500
predicted_traj[walk_start_frame:walk_end_frame, :, 0] = predicted_traj[walk_start_frame:walk_end_frame, :, 0] + x_increase
predicted_traj[walk_end_frame:,:,0] = predicted_traj[walk_end_frame:, :,0] - 300

# data = [predicted_traj, wanted_traj]
data = [wanted_traj]

# set the figure
fig = plt.figure(figsize=(30,30))
ax = p3.Axes3D(fig)

# create the lines for prediction and ground truth
# predict_lines = ax.plot(predicted_traj[1, 0, 0:1], predicted_traj[1, 0, 1:2], predicted_traj[1, 0, 2:3], 'go')[0]
wanted_lines = ax.plot(wanted_traj[1, 0, 0:1], wanted_traj[1, 0, 1:2], wanted_traj[1, 0, 2:3], 'ro')[0]
# lines = [predict_lines, wanted_lines]
lines = [wanted_lines]

ax.set_xlim3d([-1000, 1000])
ax.set_xlabel('X')

ax.set_ylim3d([-1000, 1000])
ax.set_ylabel('Y')

ax.set_zlim3d([-1000, 700])
ax.set_zlabel('Z')

ax.set_title('Human Motion Predictions')

def update_lines(frame, data, lines) :
    for line, traj in zip(lines, data):
        line.set_data(np.swapaxes(traj[frame, :, :2], 0,1)) # it wants the axises first...
        line.set_3d_properties(traj[frame, :, 2])
    return lines

#     # for line, data in zip(lines, dataLines) :
#     #     # NOTE: there is no .set_data() for 3 dim data...
#     #     line.set_data(data[0:2, :num])
#     #     line.set_3d_properties(data[2,:num])
#     # return lines

# # Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, fargs=(data, lines),
                              interval=5, blit=False)


# """
# A simple example of an animated plot... In 3D!
# """
# import numpy as np
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation

# def Gen_RandLine(length, dims=2) :
#     """
#     Create a line using a random walk algorithm

#     length is the number of points for the line.
#     dims is the number of dimensions the line has.
#     """
#     lineData = np.empty((dims, length))
#     lineData[:, 0] = np.random.rand(dims)
#     for index in range(1, length) :
#         # scaling the random numbers by 0.1 so
#         # movement is small compared to position.
#         # subtraction by 0.5 is to change the range to [-0.5, 0.5]
#         # to allow a line to move backwards.
#         step = ((np.random.rand(dims) - 0.5) * 0.1)
#         lineData[:, index] = lineData[:, index-1] + step

#     return lineData

# def update_lines(num, dataLines, lines) :
#     for line, data in zip(lines, dataLines) :
#         if num == 1:
#             print(data[0:2, :num])
#         # NOTE: there is no .set_data() for 3 dim data...
#         line.set_data(data[0:2, :num])
#         line.set_3d_properties(data[2,:num])
#     return lines

# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)

# # Fifty lines of random 3-D lines
# data = [Gen_RandLine(25, 3) for index in range(50)]

# # Creating fifty line objects.
# # NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# # Setting the axes properties
# ax.set_xlim3d([0.0, 1.0])
# ax.set_xlabel('X')

# ax.set_ylim3d([0.0, 1.0])
# ax.set_ylabel('Y')

# ax.set_zlim3d([0.0, 1.0])
# ax.set_zlabel('Z')

# ax.set_title('3D Test')

# # Creating the Animation object
# line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
#                               interval=50, blit=False)

plt.show()

    
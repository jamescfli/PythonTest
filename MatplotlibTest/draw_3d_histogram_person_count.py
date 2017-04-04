from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# read csv file
# date = '20170304'
date = '20170310'
person_count_hour_min_frame = np.loadtxt('data/person_counter_{}.txt'.format(date), delimiter=',')
print person_count_hour_min_frame.shape     # (720, 510) 510 frames per minute

# derive avg person count per minute,
person_count_hour_min = np.round(np.mean(person_count_hour_min_frame, axis=1).reshape((12, 60)))
print person_count_hour_min.shape       # from 10am to 10pm, from 1 to 60 min

# estimate # of customers, 2 min exists = one customer by LiLi
customer_counter = 0
for hour in np.arange(person_count_hour_min.shape[0]):
    for minute in np.arange(0, person_count_hour_min.shape[1], step=2):
        nb_person = min(person_count_hour_min[hour][minute], person_count_hour_min[hour][minute+1])
        if nb_person >= 2:
            customer_counter += nb_person-1
print customer_counter

# derive avg person count per hour
person_count_hour = np.mean(person_count_hour_min, axis=1)
print person_count_hour.shape

# # figure
# fig = plt.figure()
#
# ax1 = fig.add_subplot(121)
# ax1.plot(np.arange(10, 22), person_count_hour)
# ax1.set_xlim([10, 21])
#
# ax2 = fig.add_subplot(122, projection='3d')
# # create mesh, x - min, y - hour
# min_data, hour_data = np.meshgrid(np.arange(person_count_hour_min.shape[1])+1,
#                                   np.arange(person_count_hour_min.shape[0])+10)
#
# min_data = min_data.flatten()
# hour_data = hour_data.flatten()
# pc_data = person_count_hour_min.flatten()
#
# # bar3d(x, y, z, dx, dy, dz, color=None, zsort='average', *args, **kwargs)
# ax2.bar3d(min_data, hour_data, np.zeros(len(pc_data)), 0.1, 0.5, pc_data, alpha=0.7, color='r')
# plt.show()
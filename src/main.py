import numpy as np 
import cv2
from matplotlib import pyplot as plt
import pygame
from frame import *
from constants import *
from Display2D import *
from Display3D import *
import sys

# initialising displays
f2d = dim2display()
f3d = dim3display()

# pygame.init()
# screen = pygame.display.set_mode([W,H])
# capture video	
cap = cv2.VideoCapture('../test.mp4')  
s = 1

allworldCoords = []
current_frames = []
while cap.isOpened():
	
	ret, frame = cap.read()
	if (ret == False):
		break

	frame = cv2.resize(frame, (W, H))	
	frame = Frame(frame)
	current_frames.append(frame)	#list of all frames

	getFeatures(frame)	# Extract features from image

	if len(current_frames) > 1:
		# match features from the previous frame
		features1, features2, pose, matches, numMatched = matchFeatures(current_frames[-1], current_frames[-2])
		
		# propogate pose from the initial frame
		current_frames[-1].pose = np.matmul(pose, current_frames[-2].pose)
		
		# get 3D world coordinates
		worldCoords = cv2.triangulatePoints(I44[:3], pose[:3], features1.T, features2.T)

		#Filter World coordinates 
		worldCoords = np.array(worldCoords[:,(np.abs(worldCoords[3,:]) > 0.0005) & (worldCoords[2,:] > 0)])

		if worldCoords.shape[1] > 0:
			worldCoords /= worldCoords[3,:]
			allworldCoords.append(worldCoords.T)

		# Display 2D points and 3D Map
		f3d.dispAdd(current_frames, allworldCoords)
		f2d.display2D(frame.image, matches)
		my_features = f3d.features3D(current_frames, allworldCoords)
		p3d.append(my_features)

		# print(current_frames[-1].pose)

		# print(allworldCoords)

		# if worldCoords.shape[1] > 0:
			# print(worldCoords[:,0])
pcd_x = []
pcd_y = []
pcd_z = []
my_pcd = np.array(p3d)
arr_len = len(my_pcd)
last_element = my_pcd[arr_len - 1]
pcd = last_element.tolist()
for i in range(len(pcd)):
	pcd_x.append([last_element[i][0]])
	pcd_y.append([last_element[i][1]])
	pcd_z.append([last_element[i][2]])

pcd_x = np.array(pcd_x)
pcd_y = np.array(pcd_y)
pcd_z = np.array(pcd_z)
print(pcd_x)

#pcd_xx, pcd_yy, pcd_zz = np.meshgrid(pcd_x, pcd_y, pcd_z)
pcd_x = pcd_x.astype(np.float32)
pcd_y = pcd_y.astype(np.float32)
pcd_z = pcd_z.astype(np.float32)

numberOfPoints = len(pcd_y)
# print(numberOfPoints)

#blue = (pcd_y/np.max(pcd_y))
#rgb = np.hstack((np.repeat(0.01, numberOfPoints)[:, np.newaxis], np.repeat(0.01, numberOfPoints)[:, np.newaxis], blue.T))
#print(rgb[:10])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(pcd_xx, pcd_yy, color=rgb)
#encoded_colors = pypcd.encode_rgb_for_pcl((rgb * 255).astype(np.uint8))
#new_data = np.hstack((pcd_x.T, pcd_y.T, pcd_z.T, encoded_colors[:, np.newaxis]))
new_data = np.hstack((pcd_x, pcd_y, pcd_z))
print(new_data)
#new_cloud = pypcd.make_xyz_rgb_point_cloud(new_data)
new_cloud = pypcd.make_xyz_point_cloud(new_data)
#pprint.pprint(new_cloud.get_metadata())
new_cloud.save('new_cloud1.pcd')

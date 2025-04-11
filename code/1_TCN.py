"""
The purpose of the program: To perform coordinate transformation based on POS data
and generate topological relationships of images.

The elevation in POS is the actual height. The actual ground reference elevation is 35 m
(in the line parameter transformation, it is the actual elevation)
(Note: The calculation of the collinear equation is based on the flight height).
Elevation is directly defined as 80, base_h = 0

Take the projected position of the aircraft that captured the first image as the center
of the ground photography coordinate system.
The original POS number is one less than the image number. Starting from image No. 4,
the range is from 4 to 1500, totaling 1497 images.
The POS arrangement sequence is 51234. Among them, 5, 2, and 4 are short x,
and the pitch angle changes by 45 degrees relative to 5. 1 and 3 are long x,
and they represent a 45-degree roll angle change.

"""


import numpy as np
import pandas as pd
import time
from shapely.geometry import Polygon
import math
from math import sqrt
from math import sin, cos, tan, asin
from pyproj import Proj, transform
import matplotlib.pyplot as plt
import os.path


def BLH2XYZ_WGS84(B, L, H):
    # Define the WGS84 projection
    a = 6378137
    e2 = 0.0066943799901413
    N = a / sqrt(1 - e2 * sin(B * pi / 180) * sin(B * pi / 180))
    X = (N + H) * cos(B * pi / 180) * cos(L * pi / 180)
    Y = (N + H) * cos(B * pi / 180) * sin(L * pi / 180)
    Z = (N * (1 - e2) + H) * sin(B * pi / 180)

    return X,Y,Z




start_time = time.time()  # Record the start time.

wgs84 = Proj(proj='geocent', ellps='WGS84', datum='WGS84')


# Input file path
file_path = r'../input/wuhan_pos_record.txt'

# Save path
save_path = '../result'


data = pd.read_csv(file_path, sep='\s+', header=None)

num = len(data)

# 分离角点坐标
longs = data[1].values #longitude
lats= data[2].values #latitude
uav_h = data[3].values #H

roll = data[4].values #Roll angle  Omega
pitch = data[5].values #pitch angle  Phi
yaw = data[6].values #yaw angle Kappa

lens=25 #The focal length, when viewed from below, is 25mm for the first group.
pi=math.pi

#The image plane coordinates of the four corner points are respectively: The sensor size is placed in the X-Y coordinate system.
sensor_size=[23.5,15.6]

lt_y=-sensor_size[0]/2
lt_x=sensor_size[1]/2
rt_y=sensor_size[0]/2
rt_x=sensor_size[1]/2
ld_y=-sensor_size[0]/2
ld_x=-sensor_size[1]/2
rd_y=sensor_size[0]/2
rd_x=-sensor_size[1]/2



#Define corner point file
P_SET=[[0 for i in range(8)] for i in range(num)]   #Corner coordinates
center=[[0 for i in range(3)] for i in range(num)]   #The coordinates of the projection center in the ground photogrammetric coordinate system
ground_center = [[0 for i in range(2)] for i in range(num)]   #The coordinates of the projection center projected onto the ground.
l=[0 for i in range(num)]
b=[0 for i in range(num)]
temp1=[0 for i in range(3)]




for i in range(num):
    """Please note the revision."""

    # uav_h(i) = 80

    base_h = 35


    if i > num / 5 :
        lens = 35

    roll[i] = roll[i] * pi / 180
    pitch[i] = pitch[i] * pi / 180
    yaw[i] = yaw[i] * pi / 180
    l[i] = longs[i] * pi / 180
    b[i] = lats[i] * pi / 180

#Angle parameter conversion
    #The transformation from the camera-centered coordinate system to the geocentric coordinate system,
    #with the center of the projection point of the aircraft at the time of taking the first photo as the
    #center of the camera-centered coordinate system.
    Rm_e=np.array([[-sin(l[0]),cos(l[0]),0],
        [-cos(l[0])*sin(b[0]),-sin(l[0])*sin(b[0]),cos(b[0])],
        [cos(l[0])*cos(b[0]),sin(l[0])*cos(b[0]),sin(b[0])]])

# Transformation from geocentric coordinate system to local coordinate system Re_g
    p1=np.array([[cos(l[i]),-sin(l[i]),0],[sin(l[i]),cos(l[i]),0],[0,0,1]])
    p2=np.array([[cos(b[i]+pi/2),0,-sin(b[i]+pi/2)],[0,1,0],[sin(b[i]+pi/2),0,cos(b[i]+pi/2)]])
    Re_g=np.dot(p1,p2)

    #The transformation matrix from the local coordinate system to the IMU coordinate system Rg_b
    p3 = np.array([[cos(yaw[i]), -sin(yaw[i]), 0],[sin(yaw[i]), cos(yaw[i]), 0],[0, 0, 1]])
    p4 = np.array([[cos(pitch[i]), 0, sin(pitch[i])],[0, 1, 0],[-sin(pitch[i]), 0, cos(pitch[i])]])
    p5 = np.array([[1, 0, 0],[0, cos(roll[i]), -sin(roll[i])],[0, sin(roll[i]), cos(roll[i])]])
    Rg_b =p3@p4@p5


    #The IMU is transformed to the sensor coordinate system, with the eccentric angle Rb_c, and the bias angle is ignored.
    """
    ax = 0.4176;
    ay = -0.5103;
    az = 0.5;
    Rb_c = [1, -az, -ay;az, 1, -ax;ay, ax, 1];
    """
    #The sensor coordinate system is transformed to the image space coordinate system Rc_i.
    Rc_i =np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])

    #The translation from the ground control coordinate system to the image space coordinate system Q
    """
    Q = Rm_e * Re_g * Rg_b * Rb_c * Rc_i;
    """

    Q = Rm_e@Re_g@Rg_b@Rc_i
    a11 = Q[0, 0]
    a22 = Q[0, 1]
    a33 = Q[0, 2]
    b11 = Q[1, 0]
    b22 = Q[1, 1]
    b33 = Q[1, 2]
    c11 = Q[2, 0]
    c22 = Q[2, 1]
    c33 = Q[2, 2]

    #Line parameter conversion
    result = BLH2XYZ_WGS84(lats[i], longs[i], uav_h[i])
    result0 = BLH2XYZ_WGS84(lats[0], longs[0], base_h)

    temp1[0] =result[0]-result0[0]
    temp1[1] = result[1]-result0[1]
    temp1[2] = result[2]-result0[2]

    #Convert the geocentric coordinate system to the geodetic coordinate system.
    r = np.dot(Rm_e , temp1)

    Xs = r[0]
    Ys = r[1]
    Zs = r[2]
    center[i] = [Xs,Ys,Zs]

    #Calculate the coordinates of the four corner points using the collinearity equation,
    #where uav_h represents the flight altitude (height above ground).

    Xlt = (uav_h[i] - base_h) * (a11 * lt_x + b11 * lt_y - c11 * lens) / (a33 * lt_x + b33 * lt_y - c33 * lens) + Xs
    Ylt = (uav_h[i] - base_h) * (a22 * lt_x + b22 * lt_y - c22 * lens) / (a33 * lt_x + b33 * lt_y - c33 * lens) + Ys

    Xrt = (uav_h[i] - base_h) * (a11 * rt_x + b11 * rt_y - c11 * lens) / (a33 * rt_x + b33 * rt_y - c33 * lens) + Xs
    Yrt = (uav_h[i] - base_h) * (a22 * rt_x + b22 * rt_y - c22 * lens) / (a33 * rt_x + b33 * rt_y - c33 * lens) + Ys

    Xld = (uav_h[i] - base_h) * (a11 * ld_x + b11 * ld_y - c11 * lens) / (a33 * ld_x + b33 * ld_y - c33 * lens) + Xs
    Yld = (uav_h[i] - base_h) * (a22 * ld_x + b22 * ld_y - c22 * lens) / (a33 * ld_x + b33 * ld_y - c33 * lens) + Ys

    Xrd = (uav_h[i] - base_h) * (a11 * rd_x + b11 * rd_y - c11 * lens) / (a33 * rd_x + b33 * rd_y - c33 * lens) + Xs
    Yrd = (uav_h[i] - base_h) * (a22 * rd_x + b22 * rd_y - c22 * lens) / (a33 * rd_x + b33 * rd_y - c33 * lens) + Ys

    P_SET[i] = [Xlt, Ylt, Xrt, Yrt, Xrd, Yrd, Xld, Yld]

    t1 = (Xlt+Xrt+Xrd+Xld)/4
    t2 = (Ylt+Yrt+Yrd+Yld)/4
    ground_center[i] = [t1,t2]



long = int(num / 5)

#Draw an image
plt.figure(facecolor='white')


colors = {
    "yellow": [255 / 255, 255 / 255, 0 / 255],
    "green": [153 / 255, 255 / 255, 153 / 255],
    "pink": [255 / 255, 204 / 255, 204 / 255],
    "purple": [153 / 255, 153 / 255, 204 / 255]
}

for i in range(long, long * 5 ):
    x_coords = [P_SET[i][0], P_SET[i][2], P_SET[i][4], P_SET[i][6]]
    y_coords = [P_SET[i][1], P_SET[i][3], P_SET[i][5], P_SET[i][7]]

    if (long) <= i <= long * 2 - 1:
        color = colors["yellow"]
    elif (long * 3) <= i <= long * 4 - 1:
        color = colors["green"]
    elif (long * 2) <= i <= long * 3 - 1:
        color = colors["pink"]
    elif (long * 4) <= i <= long * 5 - 1:
        color = colors["purple"]

    if any(color):
        plt.fill(x_coords, y_coords, color=color)


cyan_color = [153 / 255, 255 / 255, 255 / 255]
for i in range(0, long):
    x_coords = [P_SET[i][0], P_SET[i][2], P_SET[i][4], P_SET[i][6]]
    y_coords = [P_SET[i][1], P_SET[i][3], P_SET[i][5], P_SET[i][7]]
    plt.fill(x_coords, y_coords, color=cyan_color)


for i in range(0, long * 5 ):
    plt.plot(ground_center[i][0], ground_center[i][1], marker=".", markersize=5, color=[0, 0, 0])


plt.xlabel("X/m")
plt.ylabel("Y/m")
plt.axis([-800, 500, -1100, 600])
plt.grid(True)
plt.show()

#Save the result.
with open(os.path.join(save_path,'center.txt'), 'w') as f:
    np.savetxt(f, center, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'ground_center.txt'), 'w') as f:
    np.savetxt(f, ground_center, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'corner.txt'), 'w') as f:
    np.savetxt(f, P_SET, fmt='%.8f', delimiter=' ')


end_time = time.time()
elapsed_time = end_time - start_time  # Computation time consumption
print(f"Operation time consumption：{elapsed_time} s")
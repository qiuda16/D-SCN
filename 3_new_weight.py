"""
The purpose of the program: to calculate the values of each topological factor of the main image.
"""
import os.path
import numpy as np
import pandas as pd
import time
from shapely.geometry import Polygon


start_time = time.time()


def calculate_intersection_area(polygon1, polygon2):
    poly_1 = Polygon(polygon1)
    poly_2 = Polygon(polygon2)

    intersection_polygon = poly_1.intersection(poly_2)

    return intersection_polygon.area



def calculate_area(x1, x2, x3, y1, y2, y3):
    return abs(np.linalg.det([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]) / 2)


file_path_1 = '../input/wuhan_pos_record.txt'

file_path_2 = '../result'

save_path = '../result/weight'

pos = file_path_1
data0 = pd.read_csv(pos, sep='\s+', header=None)

yaw = data0[6].values

ground_center = os.path.join(file_path_2,'ground_center.txt')   #Ground center coordinates

data1 = pd.read_csv(ground_center, sep='\s+', header=None)
X_ground_center = data1[0].values
Y_ground_center = data1[1].values


corner = os.path.join(file_path_2,'corner.txt')  #Ground corner point coordinates
data2 = pd.read_csv(corner, sep='\s+', header=None)
X_1_corner = data2[0].values
Y_1_corner = data2[1].values
X_2_corner = data2[2].values
Y_2_corner = data2[3].values
X_3_corner = data2[4].values
Y_3_corner = data2[5].values
X_4_corner = data2[6].values
Y_4_corner = data2[7].values

X_corner = np.vstack([X_1_corner, X_2_corner, X_3_corner, X_4_corner]).T
Y_corner = np.vstack([Y_1_corner, Y_2_corner, Y_3_corner, Y_4_corner]).T

center = os.path.join(file_path_2,'center.txt')  #The coordinates of the center in the air
data3 = pd.read_csv(center, sep='\s+', header=None)
X_center = data3[0].values
Y_center = data3[1].values
Z_center = data3[2].values

corner_Area_path = os.path.join(file_path_2,'overclap_Area.txt')  #The overlapping area between images
corner_Area = np.loadtxt(corner_Area_path, dtype=float)

H = 80  #Flight altitude of an aircraft
n_images = len(data2)  #Total number of images

n_num = len(data0)




# topo
Topo_path = os.path.join(file_path_2,'Topo.txt')
Topo = np.loadtxt(Topo_path, dtype=int)


number_Topo = 0
for i in range(len(Topo)):
    for j in range((i-1),len(Topo)):
        if Topo[i, j]== 1:
            number_Topo = number_Topo + 1

print(f"The number of topological：{number_Topo}")


#重叠度
area = np.zeros((n_num, n_num))# Storage of image overlap degree
area_norm = []# Store the normalized image overlap degree
area_max = 0
area_max_ij = []

for i in range(n_num):
    for j in range(i,n_num):
        if Topo[i][j]==1:

            Area_i_j = corner_Area[i][j]

            #Take the image with the smaller area between images i and j as the denominator.
            #The area of image i divides the quadrilateral into two triangles,
            #and the area of the triangles is solved using determinants.

            Area_i_1 = calculate_area(X_1_corner[i], X_2_corner[i], X_3_corner[i],
                                      Y_1_corner[i], Y_2_corner[i], Y_3_corner[i])
            Area_i_2 = calculate_area(X_1_corner[i], X_3_corner[i], X_4_corner[i],
                                      Y_1_corner[i], Y_3_corner[i], Y_4_corner[i])
            Area_i = Area_i_1 + Area_i_2

            #The area of image j
            Area_j_1 = calculate_area(X_1_corner[j], X_2_corner[j], X_3_corner[j],
                                      Y_1_corner[j], Y_2_corner[j], Y_3_corner[j])
            Area_j_2 = calculate_area(X_1_corner[j], X_3_corner[j], X_4_corner[j],
                                      Y_1_corner[j], Y_3_corner[j], Y_4_corner[j])
            Area_j = Area_j_1 + Area_j_2

            if Area_i <= Area_j:
                area[i][j] = Area_i_j / Area_j
            elif Area_i > Area_j:
                area[i][j] = Area_i_j / Area_i

            area[j][i]=0

        else:
            area[i][j] = 0
            area[j][i] = 0

# Calculate the number of areas.
number_area = 0
n = len(area)

for i in range(n):
    for j in range(i, n):
        if area[i][j] != 0:
            number_area += 1

print(f"The number of overlapping areas：{number_area}")

#  Normalization and Calculate the maximum value
for i in range(n_num):
    for j in range(i,n_num):
        if area[i][j] > area_max:
            area_max = area[i][j]
            area_max_ij = [i,j]

# Remove images with an overlap degree less than 0.25.
area__25 = area
for i in range(n_num):
    for j in range(n_num):
        if area__25[i][j] < 0.25:
            area__25[i][j] = 0

# Calculate the quantity of area_25.
number_area__25 = 0
for i in range(len(area__25)):
    for j in range(i,len(area__25)):
        if area__25[i][j] != 0:
            number_area__25 = number_area__25 + 1

print(f"The number of images with an overlap degree less than 0.25：{number_area__25}")

#  空间象限分布

yaw_E_ij = np.zeros((n_num, n_num))# Store the angle information of the image (0-180)
yaw_E_ij_360 = np.zeros((n_num, n_num))# Store the angle information of the image (0-360)
Qua = np.zeros((n_num, n_num))# Store quadrant distribution information


yaw_vec = np.array([X_center[9] - X_center[4], Y_center[9] - Y_center[4]]) # The tenth and fifth images are used as the reference.

for i in range(n_num):
    for j in range(i,n_num):
        if Topo[i][j] == 1:
            E_ij = np.array([X_center[j] - X_center[i], Y_center[j] - Y_center[i]])
            cos_theta = np.dot(yaw_vec, E_ij) / (np.linalg.norm(yaw_vec) * np.linalg.norm(E_ij))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate the angle between 0 and 180 degrees.
            angle = np.arccos(cos_theta) * 180 / np.pi
            yaw_E_ij[i, j] = angle

            # Calculate the quadrant distribution value
            Qua[i, j] = abs(2 * cos_theta ** 2 - 1)

            # Calculate the angle between 0 and 360 degrees (determine the direction through the cross product)
            cross_z = yaw_vec[0] * E_ij[1] - yaw_vec[1] * E_ij[0]
            yaw_E_ij_360[i, j] = angle if cross_z < 0 else 360 - angle

        else:
            # Set to 0 when not adjacent.
            Qua[i, j] = Qua[j, i] = 0


#Calculate the quantity of Qua.
number_Qua = 0
for i in range(len(Qua)):
    for j in range(i,len(Qua)):
        if Qua[i][j] != 0:
            number_Qua = number_Qua + 1

print(f"The number of spatial quadrant distributions：{number_Qua}")

# Calculate the base height ratio.
B_h = np.zeros((n_num, n_num)) # Storage of base height ratio
B_h_normal = np.zeros((n_num, n_num))
for i in range(n_num):
    for j in range (i,n_num):
        if Topo[i][j] == 1:
            B_h_temp= np.sqrt(((X_center[j] - X_center[i]) ** 2 + (Y_center[j] - Y_center[i]) ** 2 + (Z_center[j] - Z_center[i]) ** 2)) / H
            B_h[i,j] = B_h_temp
        else:
            B_h[i,j] = 0
            B_h[j,i] = 0

# Find the maximum value in the array and normalize it.
max_B_h = 0
for i in range(n_num):
     for j in range(i,n_num):
         if B_h[i][j] > max_B_h:
             max_B_h = B_h[i][j]

# Normalization
for i in range(n_num):
    for j in range (i,n_num):
        B_h_normal_temp=B_h[i][j]/max_B_h
        B_h_normal[i,j] = B_h_normal_temp

#Calculate the quantity of B_h.
number_B_h = 0
for i in range(len(B_h)):
    for j in range(i,len(B_h)):
        if B_h[i][j] != 0:
            number_B_h = number_B_h + 1

print(f"The number of base to height：{number_B_h}")

#Ground projection center distance
Gro_dis = np.zeros((n_num, n_num))#Storage the ground projection center distance
Gro_dis_normal = np.zeros((n_num, n_num))# Storage the ground projection center distance after H normalization
for i in range(n_num):
    for j in range (i,n_num):
        if Topo[i][j] == 1:
            Gro_dis_temp=np.sqrt(((X_ground_center[j] - X_ground_center[i]) ** 2 + (Y_ground_center[j] - Y_ground_center[i]) ** 2))
            Gro_dis[i,j] = Gro_dis_temp
            Gro_dis_normal_temp=H / (Gro_dis_temp + H)
            Gro_dis_normal[i, j] = Gro_dis_normal_temp

        else:
            Gro_dis[i,j] = 0
            Gro_dis[j,i] = 0
            Gro_dis_normal[i,j] = 0
            Gro_dis_normal[j,i] = 0

#Calculate the quantity of Gro_dis.
number_Gro_dis = 0
for i in range(len(Gro_dis)):
    for j in range(i,len(Gro_dis)):
        if Gro_dis[i][j] != 0:
            number_Gro_dis = number_Gro_dis + 1

print(f"The number of ground projection center distance：{number_Gro_dis}")


max_Gro_dis = Gro_dis[0][1]
for i in range(len(data3)):
    for j in range (i,len(data3)):
        if Gro_dis[i][j] > max_Gro_dis:
            max_Gro_dis = Gro_dis[i][j]



min_Gro_dis = Gro_dis[0][1]
for i in range(len(data3)):
    for j in range (i,len(data3)):
        if Gro_dis[i][j] < min_Gro_dis:
            min_Gro_dis = Gro_dis[i][j]

#Like the angle of intersection on a plane
ang = np.zeros((n_num, n_num))# Store the plane intersection angle like this
int_ang_cos_single = np.zeros((n_num, n_num))# Store the cosine value of the intersection angle of the plane.

for i in range(n_num):
    for j in range(i, n_num):
        if Topo[i,j] == 1:
            E_i = np.array([X_center[i] - X_ground_center[i],Y_center[i] - Y_ground_center[i],Z_center[i]])
            E_j = np.array([X_center[j] - X_ground_center[j],Y_center[j] - Y_ground_center[j],Z_center[j]])


            # Calculate the dot product and the magnitude.
            dot_product = np.dot(E_i, E_j)
            norm_i = np.linalg.norm(E_i)
            norm_j = np.linalg.norm(E_j)

            # Calculate the cosine value
            cos_theta = np.clip(dot_product / (norm_i * norm_j), -1.0, 1.0)
            int_ang_cos_single[i, j] = cos_theta
            int_ang_cos_single[j, i] = 0

            # Calculate the intersection angle (in radians)
            ang[i, j] = np.arccos(cos_theta)
            ang[j, i] = 0


        else:
            int_ang_cos_single[i, j] = 0
            int_ang_cos_single[j, i] = 0
            int_ang_cos_single[i, i] = 0
            ang[i, j] = 0
            ang[j, i] = 0


#Calculate the quantity of ang.
number_ang = 0
for i in range(len(ang)):
    for j in range(i,len(ang)):
        if ang[i][j] != 0:
            number_ang = number_ang + 1

print(f"The number of planar intersection angles like：{number_ang}")



with open(os.path.join(save_path,'area.txt'), 'w') as f:
    np.savetxt(f, area, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'area_25.txt'), 'w') as f:
    np.savetxt(f, area__25, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'Qua.txt'), 'w') as f:
    np.savetxt(f, Qua, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'B_h.txt'), 'w') as f:
    np.savetxt(f, B_h, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'Gro_dis.txt'), 'w') as f:
    np.savetxt(f, Gro_dis, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'Gro_dis_normal.txt'), 'w') as f:
    np.savetxt(f, Gro_dis_normal, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'ang.txt'), 'w') as f:
    np.savetxt(f, ang, fmt='%.8f', delimiter=' ')
with open(os.path.join(save_path,'int_ang_cos_single.txt'), 'w') as f:
    np.savetxt(f, int_ang_cos_single, fmt='%.8f', delimiter=' ')



#Calculate the weight.
long = int(n_num / 5)


weight = np.zeros((n_num, n_num))
for i in range(n_num):
    for j in range(i,n_num):
        if area[i, j] != 0:
            condition = (
                    (i < long and j < long) or
                    ((long <= i < 2 * long) and (long <= j < 2 * long)) or
                    ((2 * long <= i < 3 * long) and (2 * long <= j < 3 * long)) or
                    ((3 * long <= i < 4 * long) and (3 * long <= j < 4 * long)) or
                    ((4 * long <= i < 5 * long) and (4 * long <= j < 5 * long))
            )

            if condition:
                weight[i, j] = Topo[i, j] * Qua[i, j] * (int_ang_cos_single[i, j] + area[i, j] +int_ang_cos_single[i, j] * area[i, j] +H / (Gro_dis[i, j] + H)) / 4
            else:
                weight[i, j] = Topo[i, j] * Qua[i, j] * (int_ang_cos_single[i, j] + area[i, j] / B_h[i, j]) / 2

            weight[j, i] = weight[i, j]

#Calculate the quantity of weight
number_weight = 0
for i in range(len(corner)-1):
    for j in range(i,(len(corner)-1)):
        if weight[i][j] != 0:
            number_weight = number_weight + 1

print(f"The number of weight：{number_weight}")

#Calculate the weight_25

weight_area_25 =np.zeros((n_num, n_num))
for i in range(n_num):
    for j in range(i,n_num):
        if (
                ((i <= long) and (j <= long)) or
                ((((long + 1) <= i) and (i <= long * 2)) and (((long + 1) <= j) and (j <= long * 2))) or
                ((((long * 2 + 1) <= i) and (i <= long * 3)) and (((long * 2 + 1) <= j) and (j <= long * 3))) or
                ((((long * 3 + 1) <= i) and (i <= long * 4)) and (((long * 3 + 1) <= j) and (j <= long * 4))) or
                ((((long * 4 + 1) <= i) and (i <= long * 5)) and (((long * 4 + 1) <= j) and (j <= long * 5)))
        ):
            if area__25[i][j] != 0:
                weight_25_temp = (Topo[i][j] * Qua[i][j] *
                               (int_ang_cos_single[i][j] + area__25[i][j] + int_ang_cos_single[i][j] *
                                area__25[i][j] + H / (Gro_dis[i][j] + H)))

                weight_25_temp = weight_25_temp/4

                weight_area_25[i,j] = weight_25_temp

        else:
            if area__25[i][j] != 0:
                weight_25_temp = Topo[i][j] * Qua[i][j] * (int_ang_cos_single[i][j] + area__25[i][j] / B_h[i][j]) / 2

                weight_area_25[i,j] = weight_25_temp

#Calculate the quantity of weight_25
number_weight_25 = 0
for i in range(len(corner)-1):
    for j in range(i,(len(corner)-1)):
        if weight_area_25[i][j] != 0:
            number_weight_25 = number_weight_25 + 1

print(f"The number of weight_25：{number_weight_25}")

# Calculate weight_03
# Add 0.3 to the weight of the downward-looking images in the opposite direction.
weight_area_03 = np.zeros((n_num, n_num))

for i in range(n_num):
    for j in range(i,n_num):
        if (
                ((i <= long) and (j <= long)) or
                ((((long + 1) <= i) and (i <= long * 2)) and (((long + 1) <= j) and (j <= long * 2))) or
                ((((long * 2 + 1) <= i) and (i <= long * 3)) and (((long * 2 + 1) <= j) and (j <= long * 3))) or
                ((((long * 3 + 1) <= i) and (i <= long * 4)) and (((long * 3 + 1) <= j) and (j <= long * 4))) or
                ((((long * 4 + 1) <= i) and (i <= long * 5)) and (((long * 4 + 1) <= j) and (j <= long * 5)))
        ):

            if area[i][j] != 0:
                weight_03_temp = (Topo[i][j] * Qua[i][j] *
                               (int_ang_cos_single[i][j] + area[i][j] + int_ang_cos_single[i][j] *
                                area[i][j] + H / (Gro_dis[i][j] + H)))
                weight_03_temp = weight_03_temp / 4

                weight_area_03[i,j] = weight_03_temp

        else:
            if area[i][j] != 0:
                weight_03_temp = Topo[i][j] * Qua[i][j] * (int_ang_cos_single[i][j] + area[i][j] / B_h[i][j]) / 2
                if (i <= long) or (j <= long):
                    weight_03_temp = weight_03_temp + 0.3

                weight_area_03[i,j] = weight_03_temp


#Calculate the quantity of weight03
number_weight_03 = 0
for i in range(len(data2)-1):
    for j in range (i,(len(data2)-1)):
        if weight_area_03[i][j] != 0:
            number_weight_03 = number_weight_03 + 1

print(f"The number of weight_03：{number_weight_03}")



# Calculate weight_25
# Add 0.2 to the weight if there is a downward-looking image in the opposite direction.
weight_area_25_02 = np.zeros((n_num, n_num))

for i in range(n_num):
    for j in range(i,n_num):
        if (
                ((i <= long) and (j <= long)) or
                ((((long + 1) <= i) and (i <= long * 2)) and (((long + 1) <= j) and (j <= long * 2))) or
                ((((long * 2 + 1) <= i) and (i <= long * 3)) and (((long * 2 + 1) <= j) and (j <= long * 3))) or
                ((((long * 3 + 1) <= i) and (i <= long * 4)) and (((long * 3 + 1) <= j) and (j <= long * 4))) or
                ((((long * 4 + 1) <= i) and (i <= long * 5)) and (((long * 4 + 1) <= j) and (j <= long * 5)))
        ):
            if area__25[i][j] != 0:
                weight_25_02_temp = (Topo[i][j] * Qua[i][j] *
                                  (int_ang_cos_single[i][j] + area__25[i][j] + int_ang_cos_single[i][j] *
                                   area__25[i][j] + H / (Gro_dis[i][j] + H)))

                weight_25_02_temp = weight_25_02_temp / 4

                weight_area_25_02[i,j] = weight_25_02_temp

        else:
            if area__25[i][j] != 0:
                weight_25_02_temp = Topo[i][j] * Qua[i][j] * (int_ang_cos_single[i][j] + area__25[i][j] / B_h[i][j]) / 2
                if (i <= long) or (j <= long):
                    weight_25_02_temp = weight_25_02_temp + 0.2

                weight_area_25_02[i,j] = weight_25_02_temp

#Calculate the quantity of weight
number_weight_25_02 = 0
for i in range(n_num-1):
    for j in range(i,(n_num-1)):
        if weight_area_25_02[i][j] != 0:
            number_weight_25_02 = number_weight_25_02 + 1

print(f"The number of weight_25_02：{number_weight_25_02}")

# Calculate weight_25
# Add 0.3 to the weight if there is a downward-looking image in the opposite direction.

weight_area_25_03 =  np.zeros((n_num, n_num))

for i in range(n_num):
    for j in range(i,n_num):
        if (
                ((i <= long) and (j <= long)) or
                ((((long + 1) <= i) and (i <= long * 2)) and (((long + 1) <= j) and (j <= long * 2))) or
                ((((long * 2 + 1) <= i) and (i <= long * 3)) and (((long * 2 + 1) <= j) and (j <= long * 3))) or
                ((((long * 3 + 1) <= i) and (i <= long * 4)) and (((long * 3 + 1) <= j) and (j <= long * 4))) or
                ((((long * 4 + 1) <= i) and (i <= long * 5)) and (((long * 4 + 1) <= j) and (j <= long * 5)))
        ):
            if area__25[i][j] != 0:
                weight_25_03_temp = (Topo[i][j] * Qua[i][j] *
                                  (int_ang_cos_single[i][j] + area__25[i][j] + int_ang_cos_single[i][j] *
                                   area__25[i][j] + H / (Gro_dis[i][j] + H)))

                weight_25_03_temp = weight_25_03_temp / 4

                weight_area_25_03[i,j] = weight_25_03_temp

        else:
            if area__25[i][j] != 0:
                weight_25_03_temp = Topo[i][j] * Qua[i][j] * (int_ang_cos_single[i][j] + area__25[i][j] / B_h[i][j]) / 2
                if (i <= long) or (j <= long):
                    weight_25_03_temp = weight_25_03_temp + 0.3

                weight_area_25_03[i,j] = weight_25_03_temp

# Calculate the quantity of weight
number_weight_25_03 = 0
for i in range(n_num - 1):
    for j in range(i, (n_num - 1)):
        if weight_area_25_03[i][j] != 0:
            number_weight_25_03 = number_weight_25_03 + 1

print(f"The number of weight_25_03：{number_weight_25_03}")

# Calculate weight_25
# Add 0.5 to the weight if there is a downward-looking image in the opposite direction.

weight_area_25_05 =  np.zeros((n_num, n_num))

for i in range(n_num):
    for j in range(i,n_num):
        if (
                ((i <= long) and (j <= long)) or
                ((((long + 1) <= i) and (i <= long * 2)) and (((long + 1) <= j) and (j <= long * 2))) or
                ((((long * 2 + 1) <= i) and (i <= long * 3)) and (((long * 2 + 1) <= j) and (j <= long * 3))) or
                ((((long * 3 + 1) <= i) and (i <= long * 4)) and (((long * 3 + 1) <= j) and (j <= long * 4))) or
                ((((long * 4 + 1) <= i) and (i <= long * 5)) and (((long * 4 + 1) <= j) and (j <= long * 5)))
        ):
            if area__25[i][j] != 0:
                weight_25_05_temp = (Topo[i][j] * Qua[i][j] *
                                  (int_ang_cos_single[i][j] + area__25[i][j] + int_ang_cos_single[i][j] *
                                   area__25[i][j] + H / (Gro_dis[i][j] + H)))

                weight_25_05_temp = weight_25_05_temp / 4

                weight_area_25_05[i,j] = weight_25_05_temp

        else:
            if area__25[i][j] != 0:
                weight_25_05_temp = Topo[i][j] * Qua[i][j] * (int_ang_cos_single[i][j] + area__25[i][j] / B_h[i][j]) / 2
                if (i <= long) or (j <= long):
                    weight_25_05_temp = weight_25_05_temp + 0.5

                weight_area_25_05[i,j] = weight_25_05_temp

# Calculate the quantity of weight
number_weight_25_05 = 0
for i in range(n_num - 1):
    for j in range(i, (n_num - 1)):
        if weight_area_25_05[i][j] != 0:
            number_weight_25_05 = number_weight_25_05 + 1

print(f"The number of weight_25_05：{number_weight_25_05}")



#Save result
with open(os.path.join(save_path,'weight.txt'), 'w') as f:
    np.savetxt(f, weight, fmt='%.8f', delimiter=' ')
print(f"weight matrix has been successfully saved！")
with open(os.path.join(save_path,'weight25.txt'), 'w') as f:
    np.savetxt(f, weight_area_25, fmt='%.8f', delimiter=' ')
print(f"weight25 matrix has been successfully saved！")
with open(os.path.join(save_path,'weight03.txt'), 'w') as f:
    np.savetxt(f, weight_area_03, fmt='%.8f', delimiter=' ')
print(f"weight03 matrix has been successfully saved！")
with open(os.path.join(save_path,'weight25_02.txt'), 'w') as f:
    np.savetxt(f, weight_area_25_02, fmt='%.8f', delimiter=' ')
print(f"weight25_02 matrix has been successfully saved存！")
with open(os.path.join(save_path,'weight25_03.txt'), 'w') as f:
    np.savetxt(f, weight_area_25_05, fmt='%.8f', delimiter=' ')
print(f"weight25_03 matrix has been successfully saved！")
with open(os.path.join(save_path,'weight25_05.txt'), 'w') as f:
    np.savetxt(f, weight_area_25_05, fmt='%.8f', delimiter=' ')
print(f"weight25_05 matrix has been successfully saved！")



end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Computation time consumption
print(f"Operation time consumption：{elapsed_time} s")













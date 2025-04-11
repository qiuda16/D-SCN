import numpy as np
import pandas as pd
import time
from shapely.geometry import Polygon
import os.path



start_time = time.time()  # Record the start time.


# Define the file path
file_path = r'../result/corner.txt'

# Save path
save_path = '../result'


def calculate_intersection_area(polygon1, polygon2):
    poly_1 = Polygon(polygon1)
    poly_2 = Polygon(polygon2)

    intersection_polygon = poly_1.intersection(poly_2)

    return intersection_polygon.area


def calculate_area(x1, x2, x3, y1, y2, y3):
    # Calculate the area of a triangle using determinants.
    return abs(np.linalg.det([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]]) / 2)


data = pd.read_csv(file_path, sep='\s+', header=None)

# Separate corner point coordinates
X_1_corner = data[0].values
Y_1_corner = data[1].values
X_2_corner = data[2].values
Y_2_corner = data[3].values
X_3_corner = data[4].values
Y_3_corner = data[5].values
X_4_corner = data[6].values
Y_4_corner = data[7].values


X_corner = np.vstack([X_1_corner, X_2_corner, X_3_corner, X_4_corner]).T
Y_corner = np.vstack([Y_1_corner, Y_2_corner, Y_3_corner, Y_4_corner]).T


num_polygons = len(data)
overclap_Area = np.zeros((num_polygons,num_polygons))      # The total overlapping area
overclap_Area_1 = []                        # Specific overlapping situations (optional)
overclap_Area_2 = []





n_images = len(data)  # Total number of images
Topo = np.zeros((n_images, n_images))  # Topological relation matrix

for i in range(n_images):

    Area_i_123 = calculate_area(X_1_corner[i], X_2_corner[i], X_3_corner[i],
                                Y_1_corner[i], Y_2_corner[i], Y_3_corner[i])
    Area_i_134 = calculate_area(X_1_corner[i], X_3_corner[i], X_4_corner[i],
                                Y_1_corner[i], Y_3_corner[i], Y_4_corner[i])

    Area_i = Area_i_123 + Area_i_134

    for j in range(i, n_images):

        if i == j:
            Topo[i,j]=0
            continue

        found_overlap = False
        # Determine whether xy can fall within the max/min xy area of the reference image.
        for k in range(4):
            if (X_corner[j, k] > min(X_corner[i,])) and (X_corner[j, k] < max(X_corner[i,])) and  (
                        Y_corner[j, k] > min(Y_corner[i,])) and (Y_corner[j, k] < max(Y_corner[i,])):
                Area_i12_k = calculate_area(X_1_corner[i], X_2_corner[i], X_corner[j, k],
                                             Y_1_corner[i], Y_2_corner[i], Y_corner[j, k])
                Area_i23_k = calculate_area(X_2_corner[i], X_3_corner[i], X_corner[j, k],
                                             Y_2_corner[i], Y_3_corner[i], Y_corner[j, k])
                Area_i34_k = calculate_area(X_3_corner[i], X_4_corner[i], X_corner[j, k],
                                             Y_3_corner[i], Y_4_corner[i], Y_corner[j, k])
                Area_i41_k = calculate_area(X_4_corner[i], X_1_corner[i], X_corner[j, k],
                                             Y_4_corner[i], Y_1_corner[i], Y_corner[j, k])
                Area_i_k = Area_i12_k + Area_i23_k + Area_i34_k + Area_i41_k;
                if (Area_i_k - Area_i) < 0.01:
                    Topo[i, j] = 1
                    break

            #None of the corner points representing j are within i.
            if k == 3:
                k = k + 1

        if k == 4:
            Area_j_123 = calculate_area(X_1_corner[j], X_2_corner[j], X_3_corner[j],
                                        Y_1_corner[j], Y_2_corner[j], Y_3_corner[j])
            Area_j_134 = calculate_area(X_1_corner[j], X_3_corner[j], X_4_corner[j],
                                        Y_1_corner[j], Y_3_corner[j], Y_4_corner[j])
            Area_j = Area_j_123 + Area_j_134
            for k in range(4):
                if (X_corner[i, k] > min(X_corner[j,])) and (X_corner[i, k] < max(X_corner[j,])) and (
                        Y_corner[i, k] > min(Y_corner[j,])) and (Y_corner[i, k] < max(Y_corner[j,])):
                    Area_j12_k = calculate_area(X_1_corner[j], X_2_corner[j], X_corner[i, k],
                                                Y_1_corner[j], Y_2_corner[j], Y_corner[i, k])
                    Area_j23_k = calculate_area(X_2_corner[j], X_3_corner[j], X_corner[i, k],
                                                Y_2_corner[j], Y_3_corner[j], Y_corner[i, k])
                    Area_j34_k = calculate_area(X_3_corner[j], X_4_corner[j], X_corner[i, k],
                                                Y_3_corner[j], Y_4_corner[j], Y_corner[i, k])
                    Area_j41_k = calculate_area(X_4_corner[j], X_1_corner[j], X_corner[i, k],
                                                Y_4_corner[j], Y_1_corner[j], Y_corner[i, k])
                    Area_j_k = Area_j12_k + Area_j23_k + Area_j34_k + Area_j41_k
                    if (Area_j_k - Area_j) < 0.01:
                        Topo[i, j] = 1;
                        break

number_Topo = 0
for i in range(len(Topo)):
    for j in range((i-1),len(Topo)):
        if Topo[i, j]== 1:
            number_Topo = number_Topo + 1

#Calculate the area.

for i in range(n_images):
    for j in range((i-1), n_images):
        if Topo[i, j] == 1:
            poly1 = Polygon(zip(X_corner[i], Y_corner[i]))
            poly2 = Polygon(zip(X_corner[j], Y_corner[j]))

            overclap_Area[i, j] = calculate_intersection_area(poly1,poly2)

    for k in range (1,5):
        if i == len(data) / 5 * k:
            print(f"Calculation progress：{20 * k}%")
print(f"Calculation progress：100%")
print(f"The calculation is complete.")


with open(os.path.join(save_path,'Topo.txt'), 'w') as f:
    np.savetxt(f, Topo, fmt='%d', delimiter=' ')
with open(os.path.join(save_path,'overclap_Area.txt'), 'w') as f:
    np.savetxt(f, overclap_Area, fmt='%.8f', delimiter=' ')


end_time = time.time()
elapsed_time = end_time - start_time  # Computation time consumption
print(f"The number of topological relations：{number_Topo}")
print(f"Operation time consumption：{elapsed_time} s")

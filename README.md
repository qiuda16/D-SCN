# D-SCN
#  Project Description
The purpose of this project is to filter the input images for **3D reconstruction** in order to **reduce** redundant images to improve the reconstruction accuracy and efficiency.  
By inputting the coordinates of the pos images, the overlap, base-height ratio, topological relationship and other elements are calculated and the weight matrix is computed.  
Finally, the sparse matrix is constructed based on the weight matrix, and the matched pairs of images corresponding to the sparse matrix are directly input into COLMAP.
<div align=center><img alt="POS影像模型" height="=400" src="img/img1.png" width="400"/></div>


# Instructions for use
## Each part of the code
### 1. Data preparation
Batch read the image information and store it as a txt file in the following format.
>Image name Longitude Latitude Elevation Roll angle Pitch angle Yaw angle No.
Splitting with spaces
The example is as follows:
>4	112.458626	31.11932643	115.4455	-10.043311	-6.616909	-161.162245	1  
>5	112.4586668	31.11922387	115.4224	-8.139415	-4.966411	-159.771828	2  
>...

### 2. TCN (fully connected matrix)
The input is the data file constructed in the first step.  
The input file storage path **file_path** needs to be changed when using it
```
file_path = r'   '
```
Output as:  
>Coordinates of the center of projection in the ground photogrammetric coordinate system (center.txt),  
>Coordinates of the center of projection projected onto the ground (ground_center.txt),  
>Corner coordinates (corner.txt).

### 3. OverClap_Area (calculate overlap area)
The input is the **corner.txt** file output in step 2  
No need to modify the **file_path** when using it.
The output is:  
>Topological Relationship Matrix (Topo.txt)  
>Image Overlap Area (overclap_Area.txt)

### 4. weight (calculate weight matrix)
The input is divided into two parts:  
The first part is the data file constructed in the first step.  
To use it, you only need to change **file_path_1**.  
The second part is all the output from the second and third steps, including.
>ground_center.txt  
>corner.txt  
>center.txt  
>overclap_Area.txt  
>Topo.txt

No need to modify **file_path_2**
The output is.
>image overlap (area.txt)  
>Images with overlap greater than 0.25 (area_25.txt)  
>quadrant distribution information (qua.txt)  
>Base-to-height ratio (B_h.txt)  
>Ground projection center distance (Gro_dis.txt)  
>Normalized ground projection center distance (Gro_dis_normal.txt)  
>Image plane rendezvous angle (ang.txt)  
>cosine of the image plane rendezvous angle (int_ang_cos_single.txt)

As well as:
>Weighting Matrix

This project can output a variety of weight matrices, which can be called as needed.

### 5. Graph_SFM (construct sparse matrix)
The input is the weight matrix from step 4, which needs to be converted from **txt to csv** format
``python
if __name__ == “__main__”.
    file_path = “input path”
    out_path = “Output path”
    demension = 30 # resize the matrix accordingly
    graph_sfm(file_path, out_path, demension)
```
The output is a sparse matrix.  

**Note: The cplex package used in this project is a lightweight version that can only handle up to 1000 data.  
If you have higher needs, you can go to the IBM website to buy the full version or apply for the academic version of cplex.

```
# 6. matching_connect (matching)
Output the SCN matrix as corresponding image matching pairs, directly into colmap.

How to use the project
Follow the steps below to run it:  

cd ../code
python 1_TCN.py
python 2_OverClap_Area.py
python 3_new_weight.py
python 4_Graph_SFM_RE.py
python 5_matching_connect.py
```

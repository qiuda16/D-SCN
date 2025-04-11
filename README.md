# D-SCN
# 项目介绍
本项目旨在对**三维重建输入图像进行筛选**，从而**减少**冗余图像以提高重建精度与重建效率。  
通过输入pos影像坐标信息，计算影像间的重叠度，基高比，拓扑关系等要素并计算出weight矩阵。  
最终根据weight矩阵构建稀疏矩阵，将稀疏矩阵对应影像匹配对直接输入COLMAP。  
<div align=center><img alt="POS影像模型" height="=400" src="img/img1.png" width="400"/></div>

# 环境依赖
本项目基于**python3.9**  
所需环境请参考**requirements.txt**  
运行以下代码可在conda环境环境中直接安装
```
pip install -r requirements.txt
```


# 目录结构描述
    ├── code                        // 包含全部代码的文件夹

    │   ├── 1_TCN.py                // 计算全连接矩阵

    │   ├── 2_OverClap_Area.py      // 计算影像间重叠面积

    │   ├── 3_new_weight.py         // 通过拓扑要素矩阵计算weight矩阵

    │   ├── 4_Graph_SFM_RE.py       // 输入weight矩阵，构建稀疏矩阵

    │   └── 5_matching_connect.py   // 将SCN矩阵输出为对应影像匹配对

    ├── img                         // 存储图片的文件夹

    ├── input                       // 存储原始输入数据的文件夹

    ├── result                      // 存储结果的文件夹

    │   └── weight                  // 存储权值矩阵
    
    ├── ReadMe.md                   // 帮助文档
    
    └── requirements.txt            // 环境目录



# 使用说明
## 各部分代码明
### 1、数据准备
批量读取图片信息并存储为txt文件，存储格式如下:
>图像名 经度 纬度 高程 横滚角 俯仰角 偏航角 序号

用空格进行分割
示例如下：
>4	112.458626	31.11932643	115.4455	-10.043311	-6.616909	-161.162245	1  
>5	112.4586668	31.11922387	115.4224	-8.139415	-4.966411	-159.771828	2  
>...

### 2.TCN（全连接矩阵）
输入为第一步的构建的数据文件。  
使用时需更改输入文件存储路径**file_path**
```
file_path = r'你的文件路径'
```
输出为：  
>地面摄影测量坐标系下投影中心的坐标（center.txt）、  
>投影中心投影到地面的坐标（ground_center.txt）、  
>角点坐标（corner.txt）


### 3.OverClap_Area（计算重叠面积）
输入为第二步输出的**corner.txt**文件  
使用时无需修改**file_path**
输出为：  
>拓扑关系矩阵（Topo.txt）  
>影像重叠面积（overclap_Area.txt）

### 4.weight（计算权重矩阵）
输入分为两部分：  
第一部分为第一步的构建的数据文件。  
使用时只需更改**file_path_1**即可  
第二部分为第二步与第三步的所有输出结果，包括:
>ground_center.txt  
>corner.txt  
>center.txt  
>overclap_Area.txt  
>Topo.txt

无需修改**file_path_2**
输出结果为:
>影像重叠度（area.txt）  
>重叠度大于0.25的影像（area_25.txt）  
>象限分布信息（Qua.txt）  
>基高比（B_h.txt）  
>地面投影中心距离（Gro_dis.txt）  
>归一化后的地面投影中心距离（Gro_dis_normal.txt）  
>像平面交会角（ang.txt）  
>像平面交会角的余弦值（int_ang_cos_single.txt）

以及：
>权重矩阵

本项目可输出多种权重矩阵，可以按需进行调用。

### 5、Graph_SFM（构建稀疏矩阵）
输入为第四步的权重矩阵，需将**txt转为csv**格式
```python
if __name__ == "__main__":
    file_path = "输入路径"
    out_path = "输出路径"
    demension = 30 #按照实际调整矩阵尺寸
    graph_sfm(file_path, out_path, demension)
```
输出为稀疏矩阵。  

**注：本项目所使用cplex包为轻量版，最多仅可处理1000个数据。  
若有更高需求可去IBM官网自行购买完整版或申请学术版cplex。**


### 6.matching_connect（匹配）
将SCN矩阵输出为对应的影像匹配对，直接输入colmap即可。

## 如何使用项目
按照以下步骤运行：  
```
cd ../code
python 1_TCN.py
python 2_OverClap_Area.py
python 3_new_weight.py
python 4_Graph_SFM_RE.py
python 5_matching_connect.py
```

# 鸣谢
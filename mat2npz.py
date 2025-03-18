import os
import shutil
import scipy.io
import numpy as np

# 定义数据目录
data_dir = "data"

# 获取所有 .mat 文件
mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

# 遍历每个 .mat 文件
for mat_file in mat_files:
    dataset_name = os.path.splitext(mat_file)[0]  # 去掉 .mat 后缀
    dataset_folder = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)  # 创建对应的文件夹
    
    # 读取 .mat 文件
    mat_path = os.path.join(data_dir, mat_file)
    mat_data = scipy.io.loadmat(mat_path)
    
    # 提取需要的键值
    edge_index = mat_data.get("edge_index", None)
    feats = mat_data.get("node_feat", None)
    labels = mat_data.get("label", None)
    
    # 确保数据存在
    if edge_index is None or feats is None or labels is None:
        print(f"Warning: {mat_file} is missing required fields.")
        continue
    
    # 转换 labels 到 (n,)
    labels = np.squeeze(labels)
    
    # 保存为 .npz 文件
    npz_path = os.path.join(dataset_folder, f"{dataset_name}.npz")
    np.savez(npz_path, edge_index=edge_index, feats=feats, labels=labels)
    
    print(f"Converted {mat_file} to {npz_path}")
    print(f"Shapes - edge_index: {edge_index.shape}, feats: {feats.shape}, labels: {labels.shape}")
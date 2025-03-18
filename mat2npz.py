import os
import shutil
import scipy.io
import numpy as np

data_dir = "data"

mat_files = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

for mat_file in mat_files:
    dataset_name = os.path.splitext(mat_file)[0]  # 去掉 .mat 后缀
    dataset_folder = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)  # 创建对应的文件夹
    
    mat_path = os.path.join(data_dir, mat_file)
    mat_data = scipy.io.loadmat(mat_path)
    
    edge_index = mat_data.get("edge_index", None)
    feats = mat_data.get("node_feat", None)
    labels = mat_data.get("label", None)
    
    if edge_index is None or feats is None or labels is None:
        print(f"Warning: {mat_file} is missing required fields.")
        continue
    
    labels = np.squeeze(labels)
    
    npz_path = os.path.join(dataset_folder, f"{dataset_name}.npz")
    np.savez(npz_path, edge_index=edge_index, feats=feats, labels=labels)
    
    print(f"Converted {mat_file} to {npz_path}")
    print(f"Shapes - edge_index: {edge_index.shape}, feats: {feats.shape}, labels: {labels.shape}")
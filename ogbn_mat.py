import os
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

# 将所有需要的类添加到安全白名单
torch.serialization.add_safe_globals([
    DataEdgeAttr,
    DataTensorAttr,
    GlobalStorage
])

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

dataset_name = 'ogbn-arxiv'

# 关键修改1：指定生成COO格式的稀疏张量
dataset = PygNodePropPredDataset(
    name=dataset_name,
    transform=T.ToSparseTensor(layout=torch.sparse_coo)  # 显式指定COO布局
)

print("Dataset loaded successfully!")

data = dataset[0]

# ================== 新增分割文件生成部分 ==================
# 创建 splits 目录
splits_dir = os.path.join('data', 'splits')
os.makedirs(splits_dir, exist_ok=True)

# 获取官方分割索引
split_idx = dataset.get_idx_split()

# 转换为 numpy 格式字典
split_dict = {
    'train': split_idx['train'].numpy(),
    'valid': split_idx['valid'].numpy(),
    'test': split_idx['test'].numpy()
}

# 包装成数组（支持多组分割）
split_array = np.array([split_dict], dtype=object)

# 保存分割文件
split_filename = os.path.join(splits_dir, f'{dataset_name}-splits.npy')
np.save(split_filename, split_array, allow_pickle=True)
print(f"Saved splits to {split_filename}")
# ========================================================

# 关键修改2：正确提取COO格式索引
edge_index = data.adj_t.indices().numpy()  # 直接获取COO格式的indices

# 获取节点特征和标签，并转换为numpy数组
node_feat = data.x.numpy()
label = np.squeeze(data.y.numpy())

# 指定保存.mat文件的目录
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
mat_path = os.path.join(output_dir, f"{dataset_name}.mat")

# 保存数据
mat_dict = {
    "edge_index": edge_index,
    "node_feat": node_feat,
    "label": label
}
sio.savemat(mat_path, mat_dict)

print(f"Saved dataset to {mat_path}")
print(f"Shapes - edge_index: {edge_index.shape}, node_feat: {node_feat.shape}, label: {label.shape}")
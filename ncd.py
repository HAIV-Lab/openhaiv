import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_images_from_txt(txt_path, target_size=(64, 64)):
    """
    从TXT文件加载图片路径并处理为特征向量
    :param txt_path: 包含图片路径的TXT文件
    :param target_size: 统一调整的图片尺寸
    :return: (图片路径列表, 特征矩阵)
    """
    image_paths = []
    features = []
    
    with open(txt_path, 'r') as f:
        for line in f:
            path = line.strip()  # 移除换行符和空格
            if not path or not os.path.exists(path):
                print(f"警告：路径不存在或为空 -> {path}")
                continue
                
            try:
                # 打开图片并转换为特征向量
                img = Image.open(path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img).flatten()  # 展平为1D向量
                features.append(img_array)
                image_paths.append(path)
            except Exception as e:
                print(f"处理图片失败 {path}: {str(e)}")
    
    if not image_paths:
        raise ValueError("未找到有效图片路径，请检查TXT文件内容")
    
    return image_paths, np.array(features)

# 1. 从TXT文件加载图片
txt_file = "/data/zqh/xz/openhaiv-ood_xz/output/ood/msp_cil_test/exp/0.9995ood.txt"  # 替换为你的TXT文件路径
image_paths, X = load_images_from_txt(txt_file, target_size=(128, 128))  # 增大尺寸保留更多特征

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 执行K-means聚类
n_clusters = 95  # 根据实际情况调整
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # 显式设置n_init避免警告
clusters = kmeans.fit_predict(X_scaled)

# 4. 创建输出文件夹
output_dir = "clustered_images"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 5. 复制图片到对应聚类文件夹
for cluster_id in range(n_clusters):
    cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    cluster_indices = np.where(clusters == cluster_id)[0]
    for idx in cluster_indices:
        src_path = image_paths[idx]
        # 生成简单文件名：cluster_X_img_序号.扩展名
        filename = f"cluster_{cluster_id}_img_{idx}{os.path.splitext(src_path)[1]}"
        dest_path = os.path.join(cluster_dir, filename)
        
        shutil.copy2(src_path, dest_path)

print(f"聚类完成！共处理 {len(image_paths)} 张图片")
print(f"聚类分布: {np.bincount(clusters)}")
print(f"结果保存在: {os.path.abspath(output_dir)}")
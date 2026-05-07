import numpy as np
import matplotlib.pyplot as plt

# 如果数据来自 txt 文件，可以用这个读取
scores = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-08-22_S_query_I_confid_141/prediction/141/000300.txt")

# 直接用你提供的 anomaly score 列表
#scores = np.array([
#    0.001058, 0.001359, 0.001807, 0.000861, 0.000586, 0.001349,
#    0.000522, 0.000704, 0.001023, 0.002047, 0.011482, 0.001159,
#    0.001409, 0.001502, 0.001490
#])

# 统计分析
max_val = np.max(scores)
min_val = np.min(scores)
mean_val = np.mean(scores)
median_val = np.median(scores)
std_val = np.std(scores)

print("Anomaly Score Analysis:")
print(f"Max value     : {max_val:.6f}")
print(f"Min value     : {min_val:.6f}")
print(f"Mean          : {mean_val:.6f}")
print(f"Median        : {median_val:.6f}")
print(f"Std Deviation : {std_val:.6f}")

# 可视化直方图
plt.figure()
plt.hist(scores, bins=10, edgecolor='black')
plt.title("Anomaly Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


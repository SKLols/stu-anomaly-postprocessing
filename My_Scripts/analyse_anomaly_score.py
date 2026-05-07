import numpy as np
import matplotlib.pyplot as plt

# 如果数据来自 txt 文件，可以用这个读取
scores = np.loadtxt("/media/ubuntu22/HDD22T/_Sourabh/stu_dataset/Mask4Former3D/saved/2025-09-18_150218/prediction_maxlogit_oasc/125/000140.txt")

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

# Additional statistics
q1 = np.percentile(scores, 25)
q3 = np.percentile(scores, 75)
iqr = q3 - q1
variance = np.var(scores)
total_points = len(scores)

print(f"\nAdditional Statistics:")
print(f"Q1 (25th percentile) : {q1:.6f}")
print(f"Q3 (75th percentile) : {q3:.6f}")
print(f"IQR                  : {iqr:.6f}")
print(f"Variance             : {variance:.6f}")
print(f"Total points         : {total_points}")

# Count extreme values
threshold_low = -10
threshold_high = 5
extreme_low_count = np.sum(scores < threshold_low)
extreme_high_count = np.sum(scores > threshold_high)

print(f"\nExtreme Value Counts:")
print(f"Points < {threshold_low} : {extreme_low_count} ({extreme_low_count/total_points*100:.2f}%)")
print(f"Points > {threshold_high} : {extreme_high_count} ({extreme_high_count/total_points*100:.2f}%)")


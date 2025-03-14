import torch

import numpy as np
import torch

# 创建一个 20x20 的随机矩阵
matrix_np = np.random.rand(20, 20)

# 计算要置零的元素个数，总元素个数的60%
num_zeros = int(0.6 * 20 * 20)

# 将随机选择的元素置零
random_indices = np.random.choice(20 * 20, num_zeros, replace=False)
row_indices, col_indices = np.unravel_index(random_indices, (20, 20))
matrix_np[row_indices, col_indices] = 0

# 将 NumPy 数组转换为 PyTorch 张量
matrix = torch.tensor(matrix_np)
matrix[matrix != 0] = 1.0

# 打印结果
print("Original matrix:")
print(matrix)


# 进行奇异值分解
U, S, V = torch.svd(matrix)

# 取前k个奇异值进行低秩近似
k = 10
matrix_approx = torch.mm(U[:, :k] * S[:k], V[:, :k].t())

# 输出分解后的小矩阵
print("Approximated matrix:")
print(matrix_approx)

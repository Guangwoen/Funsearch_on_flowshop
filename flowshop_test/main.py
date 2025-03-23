import numpy as np
from itertools import permutations

# 输入数据矩阵（工件×机器）
processing_times = np.array([
    [375, 12, 142, 245, 412],
    [632, 452, 758, 278, 398],
    [12, 876, 124, 534, 765],
    [460, 542, 523, 120, 499],
    [528, 101, 789, 124, 999],
    [796, 245, 632, 375, 123],
    [532, 230, 543, 896, 452],
    [14, 124, 214, 543, 785],
    [257, 527, 753, 210, 463],
    [896, 896, 214, 258, 259],
    [532, 302, 501, 765, 988]
])

num_jobs, num_machines = processing_times.shape

# Step 1: 计算每个工件的总处理时间，并按降序排序
total_processing_times = np.sum(processing_times, axis=1)
sorted_jobs = np.argsort(-total_processing_times)  # 降序排列的工件索引


# 计算Flowshop调度的完工时间（Cmax）
def calculate_makespan(job_sequence, processing_times):
    num_jobs = len(job_sequence)
    num_machines = processing_times.shape[1]

    C = np.zeros((num_jobs, num_machines))  # 完工时间矩阵

    for i, job in enumerate(job_sequence):
        for m in range(num_machines):
            if i == 0 and m == 0:
                C[i, m] = processing_times[job, m]
            elif i == 0:
                C[i, m] = C[i, m - 1] + processing_times[job, m]
            elif m == 0:
                C[i, m] = C[i - 1, m] + processing_times[job, m]
            else:
                C[i, m] = max(C[i - 1, m], C[i, m - 1]) + processing_times[job, m]

    return C[-1, -1]  # 返回最终完工时间


# Step 2: 使用NEH算法构造最优序列
schedule = [sorted_jobs[0]]  # 先把最大工件加入调度

for job in sorted_jobs[1:]:  # 依次插入剩余工件
    best_seq = None
    best_makespan = float('inf')

    for i in range(len(schedule) + 1):  # 遍历所有插入位置
        new_seq = schedule[:i] + [job] + schedule[i:]
        makespan = calculate_makespan(new_seq, processing_times)

        if makespan < best_makespan:
            best_makespan = makespan
            best_seq = new_seq

    schedule = best_seq  # 更新调度序列

# 计算最终的完工时间
final_makespan = calculate_makespan(schedule, processing_times)

# 返回最终调度顺序及其完工时间
schedule, final_makespan

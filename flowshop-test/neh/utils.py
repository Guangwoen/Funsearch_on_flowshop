import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_datasets(dataset_folder):
    datasets = {}

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    first_line = f.readline().strip().split()
                    n_jobs, n_machines = int(first_line[0]), int(first_line[1])

                    raw_data = np.loadtxt(f)
                    jobs = raw_data[:, 1::2]

                    if jobs.shape != (n_jobs, n_machines):
                        print(f"Warning: Mismatch in expected dimensions "
                              f"for {file_path}, skipping dataset.")
                        continue

                    datasets[file] = jobs

    return datasets


def plot_gantt_chart(job_sequence, processing_times):
    num_jobs = len(job_sequence)
    num_machines = processing_times.shape[1]

    # 计算任务调度时间
    C = np.zeros((num_jobs, num_machines))  # 完工时间矩阵
    start_times = np.zeros((num_jobs, num_machines))  # 任务开始时间

    for i, job in enumerate(job_sequence):
        for m in range(num_machines):
            if i == 0 and m == 0:
                start_times[i, m] = 0
                C[i, m] = processing_times[job, m]
            elif i == 0:
                start_times[i, m] = C[i, m - 1]
                C[i, m] = start_times[i, m] + processing_times[job, m]
            elif m == 0:
                start_times[i, m] = C[i - 1, m]
                C[i, m] = start_times[i, m] + processing_times[job, m]
            else:
                start_times[i, m] = max(C[i - 1, m], C[i, m - 1])
                C[i, m] = start_times[i, m] + processing_times[job, m]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Paired.colors  # 颜色池

    for i, job in enumerate(job_sequence):
        color = colors[i % len(colors)]
        for m in range(num_machines):
            start = start_times[i, m]
            duration = processing_times[job, m]
            rect = patches.Rectangle((start, m), duration, 0.8, edgecolor='black', facecolor=color,
                                     label=f"Job {job}" if m == 0 else "")
            ax.add_patch(rect)
            ax.text(start + duration / 2, m + 0.4, f"J{job}", ha='center', va='center', fontsize=10, color='black')

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)])
    ax.set_title("Gantt Chart for NEH Scheduling")
    ax.set_xlim(0, np.max(C) + 50)
    ax.set_ylim(-0.5, num_machines - 0.5)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


def calc_makespan(job_seq, proc_times):
    num_jobs = len(job_seq)
    num_machines = proc_times.shape[1]

    C = np.zeros((num_jobs, num_machines))

    for i, job in enumerate(job_seq):
        for m in range(num_machines):
            if i == 0 and m == 0:
                C[i, m] = proc_times[job, m]
            elif i == 0:
                C[i, m] = C[i, m-1] + proc_times[job, m]
            elif m ==0:
                C[i, m] = C[i-1, m] + proc_times[job, m]
            else:
                C[i, m] = max(float(C[i-1, m]), float(C[i, m-1])) + proc_times[job, m]

    return C[-1, -1]


def extract_function_from_json(json_file: str, output_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    function_code = data.get("function", "")

    if not function_code:
        print("No function code found in JSON.")
        return

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(function_code)

    print(f"Function code written to {output_file}")


if __name__ == '__main__':
    os.chdir('/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop')
    extract_function_from_json('flowshop/logs/evaluator_log/samples/samples_21.json', 'flowshop-test/neh/evolved.py')


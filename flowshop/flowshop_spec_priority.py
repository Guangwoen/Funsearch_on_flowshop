import numpy as np


# 计算调度顺序的makespan
def calculate_makespan(order: list[int], proc_time: np.ndarray) -> int:
    num_machines = len(proc_time[0])
    num_jobs = len(order)
    machine_times = [[0] * (num_jobs + 1) for _ in range(num_machines)]

    for i in range(num_jobs):
        job_id = order[i]
        times = proc_time[job_id]
        machine_times[0][i + 1] = machine_times[0][i] + times[0]
        for m in range(1, num_machines):
            machine_times[m][i + 1] = max(machine_times[m][i], machine_times[m - 1][i + 1]) + times[m]

    return machine_times[-1][-1]


def order_schedule(jobs: np.ndarray) -> list[int]:
    remaining_jobs = list(enumerate(jobs))
    schedule = []

    while remaining_jobs:
        # 计算每个作业的优先级
        priorities = []
        for idx, job in remaining_jobs:
            priority = calc_priority(job)
            priorities.append((priority, -job[0], idx, job))

        # 按优先级降序排序
        priorities.sort(reverse=True, key=lambda x: (x[0], x[1]))

        # 选择优先级最高的作业
        selected = priorities[0]
        schedule.append(selected[2])
        remaining_jobs = [(idx, job) for idx, job in remaining_jobs if idx != selected[2]]

    return schedule


@funsearch.run
def evaluate(instances: dict) -> float:
    """
    FunSearch evaluation function that computes the average makespan across multiple datasets.
    - instances: dict mapping instance names to 2D numpy arrays (processing time matrices).
    Returns the negative mean makespan (float) for optimization.
    """
    makespans = []
    for name in instances:
        processing_times = instances[name]
        if not isinstance(processing_times, np.ndarray):
            print(f"[ERROR] Instance {name} is not ndarray")
            continue
        if not np.issubdtype(processing_times.dtype, np.integer):
            processing_times = processing_times.astype(int)

        schedule = order_schedule(processing_times)
        ms = calculate_makespan(schedule, processing_times)
        makespans.append(ms)

    if not makespans:
        return 1e9
    return -float(np.mean(makespans))


@funsearch.evolve
def calc_priority(job: np.ndarray) -> float:
    first_machine_time = job[0]

    # Total processing time (smaller is better)
    total_time = sum(job)

    # Calculate priority score (using negative values since we're sorting in reverse)
    # We can adjust weights of these factors based on problem characteristics
    w1 = 0.7  # Weight for first machine time
    w2 = 0.3  # Weight for total processing time

    # Calculate a composite priority value
    # Higher priority means this job should be processed earlier
    # Since we're using reverse=True in sorting, we negate these values
    # so that smaller processing times get higher priority
    priority = -(w1 * first_machine_time + w2 * total_time)

    return priority


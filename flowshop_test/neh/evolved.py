import numpy as np

from flowshop_test.utils import calc_makespan


def compute_makespan(schedule: list[int], processing_times: np.ndarray) -> int:
    """
    Compute the makespan (total completion time) for a given job schedule in a PFSP.
    - schedule: list of job indices in the order they are processed.
    - processing_times: 2D numpy array of shape (num_jobs, num_machines) with processing times for each job on each machine.
    Returns the makespan (int) for the given order.
    """
    num_jobs = len(schedule)
    num_machines = processing_times.shape[1]
    if num_jobs == 0:
        return 0

    completion_times = np.zeros((num_jobs, num_machines), dtype=int)
    first_job = schedule[0]
    completion_times[0, 0] = processing_times[first_job, 0]
    for m in range(1, num_machines):
        completion_times[0, m] = completion_times[0, m - 1] + processing_times[first_job, m]

    for i in range(1, num_jobs):
        job = schedule[i]
        completion_times[i, 0] = completion_times[i - 1, 0] + processing_times[job, 0]
        for m in range(1, num_machines):
            completion_times[i, m] = max(completion_times[i, m - 1], completion_times[i - 1, m]) + processing_times[
                job, m]

    return int(completion_times[-1, -1])


def evolved_neh(processing_times: np.ndarray) -> list[int]:
    import random
    num_jobs, num_machines = processing_times.shape

    def compute_priority_scores():
        scores = []
        weights = np.linspace(1.5, 0.5, num=num_machines)
        weighted_sums = processing_times @ weights
        for j in range(num_jobs):
            bottleneck = np.max(processing_times[j])
            score = 0.7 * weighted_sums[j] + 0.2 * processing_times[j].sum() + 0.1 * bottleneck
            scores.append((j, score))
        return sorted(scores, key=lambda x: -x[1])

    def dynamic_insertion(seq, job_id):
        best_seq = None
        best_makespan = float('inf')
        for i in range(len(seq) + 1):
            candidate = seq[:i] + [job_id] + seq[i:]
            ms = compute_makespan(candidate, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_seq = candidate
        return best_seq

    def balance_machine_load(sequence):
        loads = np.zeros((num_machines,))
        for job in sequence:
            loads += processing_times[job]
        return np.std(loads)

    def tabu_local_search(init_seq, tabu_tenure=5, max_iter=100):
        current_seq = init_seq[:]
        best_seq = current_seq[:]
        best_makespan = compute_makespan(best_seq, processing_times)
        tabu_list = {}
        iteration = 0

        while iteration < max_iter:
            neighborhood = []
            for i in range(num_jobs):
                for j in range(i + 1, num_jobs):
                    if (i, j) in tabu_list and tabu_list[(i, j)] > iteration:
                        continue
                    temp_seq = current_seq[:]
                    temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
                    ms = compute_makespan(temp_seq, processing_times)
                    load_dev = balance_machine_load(temp_seq)
                    score = ms + 0.01 * load_dev
                    neighborhood.append((score, ms, (i, j), temp_seq))

            if not neighborhood:
                break

            neighborhood.sort()
            score, ms, move, candidate_seq = neighborhood[0]
            current_seq = candidate_seq[:]
            tabu_list[move] = iteration + tabu_tenure

            if ms < best_makespan:
                best_makespan = ms
                best_seq = current_seq[:]

            iteration += 1

        return best_seq

    def adaptive_restart(base_seq, num_restarts=4):
        best_seq = base_seq[:]
        best_ms = compute_makespan(best_seq, processing_times)
        for r in range(num_restarts):
            shuffled = base_seq[:]
            random.shuffle(shuffled)
            evolved = tabu_local_search(shuffled, max_iter=30)
            ms = compute_makespan(evolved, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_seq = evolved
        return best_seq

    scored_jobs = compute_priority_scores()
    ordered_jobs = [j for j, _ in scored_jobs]

    sequence = []
    for job in ordered_jobs:
        sequence = dynamic_insertion(sequence, job)

    sequence = tabu_local_search(sequence)
    sequence = adaptive_restart(sequence, num_restarts=5)

    return sequence


from typing import List
import numpy as np


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

        schedule = neh_heuristic(processing_times)
        ms = compute_makespan(schedule, processing_times)
        makespans.append(ms)

    if not makespans:
        return 1e9
    return -float(np.mean(makespans))


@funsearch.evolve
def neh_heuristic(processing_times: np.ndarray) -> list[int]:
    """
    An enhanced initial heuristic for the Permutation Flowshop Scheduling Problem (PFSP).

    This heuristic combines:
    - A weighted scoring for each job based on its total processing time and its maximum processing time.
      The weight parameter alpha balances these two criteria.
    - An iterative insertion procedure that builds an initial sequence.
    - A subsequent local search using pairwise swap improvements to further reduce the makespan.

    The resulting schedule (a list of job indices) is returned.
    """
    num_jobs, num_machines = processing_times.shape
    alpha = 0.7  # Weight parameter: can be tuned/evolved (alpha in [0, 1])

    # Compute a weighted score for each job.
    # Lower score indicates a job should be scheduled earlier.
    job_scores = []
    for job in range(num_jobs):
        total_time = processing_times[job].sum()
        max_time = processing_times[job].max()
        score = alpha * total_time + (1 - alpha) * max_time
        job_scores.append((job, score))

    # Sort jobs by ascending score (best candidate first)
    job_scores.sort(key=lambda x: x[1])

    # Build an initial sequence using iterative insertion
    sequence = [job_scores[0][0]]
    for job, _ in job_scores[1:]:
        best_sequence = None
        best_makespan = float('inf')
        # Try inserting the job in every possible position
        for pos in range(len(sequence) + 1):
            candidate_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = compute_makespan(candidate_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_sequence = candidate_seq
        sequence = best_sequence

    # Local search: try pairwise swaps to further improve the sequence
    improvement = True
    while improvement:
        improvement = False
        current_makespan = compute_makespan(sequence, processing_times)
        for i in range(num_jobs - 1):
            for j in range(i + 1, num_jobs):
                new_seq = sequence.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_makespan = compute_makespan(new_seq, processing_times)
                if new_makespan < current_makespan:
                    sequence = new_seq
                    current_makespan = new_makespan
                    improvement = True
                    # Break out to restart the search after any improvement
                    break
            if improvement:
                break

    return sequence
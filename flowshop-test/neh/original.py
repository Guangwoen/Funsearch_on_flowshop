import numpy as np

from utils import calc_makespan


def neh(processing_times: np.ndarray) -> list[int]:
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
            ms = calc_makespan(candidate_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_sequence = candidate_seq
        sequence = best_sequence

    # Local search: try pairwise swaps to further improve the sequence
    improvement = True
    while improvement:
        improvement = False
        current_makespan = calc_makespan(sequence, processing_times)
        for i in range(num_jobs - 1):
            for j in range(i + 1, num_jobs):
                new_seq = sequence.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_makespan = calc_makespan(new_seq, processing_times)
                if new_makespan < current_makespan:
                    sequence = new_seq
                    current_makespan = new_makespan
                    improvement = True
                    # Break out to restart the search after any improvement
                    break
            if improvement:
                break

    return sequence

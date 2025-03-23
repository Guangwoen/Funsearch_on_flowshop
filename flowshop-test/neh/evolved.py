import numpy as np

from utils import calc_makespan


def evolved_neh(processing_times: np.ndarray) -> list[int]:
    """
    An enhanced initial heuristic for the Permutation Flowshop Scheduling Problem (PFSP).

    This heuristic combines:
    - A weighted scoring for each job based on its total processing time and its maximum processing time.
      The weight parameter alpha balances these two criteria.
    - An iterative insertion procedure that builds an initial sequence.
    - A subsequent local search using pairwise swap improvements to further reduce the makespan.

    The resulting schedule (a list of job indices) is returned.
    """
    """
    An improved heuristic for the Permutation Flowshop Scheduling Problem (PFSP) that minimizes makespan
    by using a modified job ordering and insertion strategy.

    The heuristic performs the following steps:
    - Orders jobs based on their maximum processing time across all machines.
    - Builds an initial sequence using a modified greedy insertion strategy.
    - Applies a local search with pairwise swaps to optimize the sequence further.

    The resulting schedule (a list of job indices) is returned.
    """
    num_jobs, num_machines = processing_times.shape

    # Step 1: Order jobs based on their maximum processing time across all machines
    job_indices = np.arange(num_jobs)
    job_order = job_indices[np.argsort(-processing_times.max(axis=1))].tolist()

    # Step 2: Build an initial sequence using a modified greedy insertion strategy
    sequence = []
    for job in job_order:
        best_position = 0
        best_makespan = float('inf')
        
        # Try inserting the job in every possible position
        for pos in range(len(sequence) + 1):
            candidate_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = calc_makespan(candidate_seq, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_position = pos
        
        # Insert the job at the best position found
        sequence.insert(best_position, job)

    # Step 3: Local search: try pairwise swaps to further improve the sequence
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


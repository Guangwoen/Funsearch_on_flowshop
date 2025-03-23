import numpy as np


def evolved_priority(job: np.ndarray) -> float:
    """Improved version of `calc_priority_v0` with more complex logic."""
    total_time = 0.0
    num_jobs = len(job)
    
    # Calculate average time to identify priority
    average_time = np.mean(job)
    
    # Initialize priority score
    priority_score = 0.0
    
    for i in range(num_jobs):
        if job[i] < average_time:
            priority_score += (average_time - job[i]) * 1.5  # Higher weight for jobs below average
        elif job[i] == average_time:
            priority_score += average_time  # Equal weight for average jobs
        else:
            priority_score += (job[i] - average_time) * 0.5  # Lower weight for jobs above average
    
    # Adding a penalty for jobs that exceed a certain threshold
    threshold = np.max(job) * 0.8
    for time in job:
        if time > threshold:
            priority_score -= (time - threshold) * 2  # Hefty penalty for exceeding threshold
    
    # Ensure priority score is non-negative
    if priority_score < 0:
        priority_score = 0
    
    total_time = sum(job) + priority_score
    return total_time


def evolved_priority_2(job: np.ndarray) -> float:
    total_time = sum(job)
    num_jobs = len(job)

    # Initialize variables for different metrics
    first_machine_time = job[0]
    last_machine_time = job[-1]
    max_time = max(job) if num_jobs > 0 else 0
    min_time = min(job) if num_jobs > 0 else 0
    avg_time = total_time / num_jobs if num_jobs > 0 else 0

    # Initialize a priority score
    priority = 0

    # Incorporate conditions based on job characteristics
    if num_jobs > 0:
        if first_machine_time < avg_time:
            priority += 1.5 * first_machine_time
        else:
            priority += 0.5 * first_machine_time

        if last_machine_time > avg_time:
            priority += 2 * (last_machine_time - avg_time)

        # Loop through all jobs to adjust priority based on variance
        for job_time in job:
            if job_time < avg_time:
                priority += (avg_time - job_time) * 0.3
            elif job_time > avg_time:
                priority -= (job_time - avg_time) * 0.2

    # Adjust priority based on max and min times
    if max_time > 0:
        priority += 0.1 * (max_time - min_time)

    return priority


def evolved_priority_3(job: np.ndarray) -> float:
    """Improved version of `calc_priority_v1` with more complexity and additional factors."""
    first_machine_time = job[0]
    total_time = sum(job)
    num_machines = len(job)

    # Initialize priority components with dynamic weights
    w1 = 0.3 + (0.1 * (first_machine_time / 10))  # Adjust weight based on first machine time
    w2 = 0.25 + (0.05 * (total_time / 100))  # Adjust weight based on total processing time
    w3 = 0.2  # Weight for number of machines involved
    w4 = 0.25  # Weight for job variability

    # Calculate machine utilization
    machine_utilization = np.zeros(num_machines)
    for time in job:
        for i in range(num_machines):
            if time > 0:
                machine_utilization[i] += 1

    avg_utilization = np.mean(machine_utilization)
    utilization_factor = 1 / (1 + avg_utilization)

    # Calculate job variability
    job_variability = np.std(job)

    # Calculate composite priority value with additional factors
    priority = -(w1 * first_machine_time + w2 * total_time + w3 * utilization_factor + w4 * job_variability)

    # Conditional adjustments based on total time and first machine time
    if total_time > 200:  # Higher threshold for total time
        priority *= 1.5  # Significantly increase priority if total time is very high
    elif total_time < 30:  # Lower threshold for total time
        priority *= 0.7  # Decrease priority if total time is very low

    if first_machine_time < 3:  # Very low first machine time
        priority *= 0.75  # Decrease priority significantly
    elif first_machine_time > 25:  # High first machine time
        priority *= 1.2  # Slightly increase priority

    # Loop through job times to apply penalties and bonuses
    for time in job:
        if time > 30:  # Penalty for very long jobs
            priority += 0.15 * (time - 30)
        elif time < 5:  # Bonus for very short jobs
            priority -= 0.1 * (5 - time)

    # Additional adjustment based on the variance of job times
    if job_variability > 10:  # High variability in job times
        priority *= 0.9  # Decrease priority due to inconsistency
    elif job_variability < 2:  # Low variability in job times
        priority *= 1.1  # Increase priority for consistency

    # Final adjustment based on total number of jobs and their distribution
    if len(job) > 6:  # More than 6 jobs
        priority *= 1.1  # Increase priority slightly
    elif len(job) < 4:  # Less than 4 jobs
        priority *= 0.95  # Decrease priority slightly

    # Extra complexity: Adjust based on the sum of the squares of job times
    sum_of_squares = sum(time ** 2 for time in job)
    if sum_of_squares > 500:  # High sum of squares
        priority *= 1.05  # Slightly increase priority
    elif sum_of_squares < 100:  # Low sum of squares
        priority *= 0.95  # Slightly decrease priority

    return priority

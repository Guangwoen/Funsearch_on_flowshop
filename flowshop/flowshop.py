import time
import json
import multiprocessing
from typing import Collection, Any
import http.client
from implementation import sampler


def _trim_preface_of_body(sample: str) -> str:
    """Trim the redundant descriptions/symbols/'def' declaration before the function body.
    Please see my comments in sampler.LLM (in sampler.py).
    Since the LLM used in this file is not a pure code completion LLM, this trim function is required.

    -Example sample (function & description generated by LLM):
    -------------------------------------
    This is the optimized function ...
    def priority_v2(...) -> ...:
        return ...
    This function aims to ...
    -------------------------------------
    -This function removes the description above the function's signature, and the function's signature.
    -The indent of the code is preserved.
    -Return of this function:
    -------------------------------------
        return ...
    This function aims to ...
    -------------------------------------
    """
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False
    for lineno, line in enumerate(lines):
        # find the first 'def' statement in the given code
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break
    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno + 1:]:
            code += line + '\n'
        return code
    return sample


class LLMAPI(sampler.LLM):
    """Language model that predicts continuation of provided source code.
    """

    def __init__(self, samples_per_prompt: int, trim=True):
        super().__init__(samples_per_prompt)
        additional_prompt = ("Improve the scheduling heuristic to minimize makespan. "
                             "You can change how jobs are ordered or inserted. "
                             "Be creative. Think beyond NEH logic. "
                             "Pls only generate neh_heuristic(processing_times: np.ndarray) function"
                             "Use loops, conditionals, or clustering ideas. Only return valid Python code.")

        self._additional_prompt = additional_prompt
        self._trim = trim

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]

    def _draw_sample(self, content: str) -> str:
        prompt = '\n'.join([content, self._additional_prompt])
        message = [{'role': 'user', 'content': prompt}]

        while True:
            try:
                conn = http.client.HTTPSConnection("api.bltcy.ai")
                payload = json.dumps({
                    "max_tokens": 1024,
                    "model": "chatgpt-4o-latest",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': 'Bearer sk-OmRJlpj2aI4A3GLvA4Bd841fCfB04b3e9eF6D0D9984f1719',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request("POST", "/v1/chat/completions", payload, headers)
                res = conn.getresponse()
                data = res.read().decode("utf-8")
                data = json.loads(data)
                response = data['choices'][0]['message']['content']
                if self._trim:
                    response = _trim_preface_of_body(response)
                return response
            except Exception:
                time.sleep(2)
                continue


from implementation import evaluator
from implementation import evaluator_accelerate


class Sandbox(evaluator.Sandbox):
    """Sandbox for executing generated code. Implemented by RZ.

    RZ: Sandbox returns the 'score' of the program and:
    1) avoids the generated code to be harmful (accessing the internet, take up too much RAM).
    2) stops the execution of the code in time (avoid endless loop).
    """

    def __init__(self, verbose=False, numba_accelerate=True):
        """
        Args:
            verbose         : Print evaluate information.
            numba_accelerate: Use numba to accelerate the evaluation. It should be noted that not all numpy functions
                              support numba acceleration, such as np.piecewise().
        """
        self._verbose = verbose
        self._numba_accelerate = False

    def run(
            self,
            program: str,
            function_to_run: str,  # RZ: refers to the name of the function to run (e.g., 'evaluate')
            function_to_evolve: str,  # RZ: accelerate the code by decorating @numba.jit() on function_to_evolve.
            inputs: Any,  # refers to the dataset
            test_input: str,  # refers to the current instance
            timeout_seconds: int,
            **kwargs  # RZ: add this
    ) -> tuple[Any, bool]:
        """Returns `function_to_run(test_input)` and whether execution succeeded.

        RZ: If the generated code (generated by LLM) is executed successfully,
        the output of this function is the score of a given program.
        RZ: PLEASE NOTE THAT this SandBox is only designed for bin-packing problem.
        """
        dataset = {test_input: inputs[test_input]}
        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=self._compile_and_run_function,
                args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
            )
            process.start()
            process.join(timeout=timeout_seconds)
            if process.is_alive():
                # if the process is not finished in time, we consider the program illegal
                process.terminate()
                process.join()
                results = None, False
            else:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                else:
                    results = None, False

            return results
        except:
            return None, False

    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, dataset, numba_accelerate,
                                  result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            # compile the program, and maps the global func/var/class name to its address
            all_globals_namespace = {}
            # execute the program, map func/var/class to global namespace
            exec(program, all_globals_namespace)
            # get the pointer of 'function_to_run'
            function_to_run = all_globals_namespace[function_to_run]
            # return the execution results
            results = function_to_run(dataset)
            # the results must be int or float
            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return
            result_queue.put((results, True))
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[Sandbox Error] {error_msg}")  # 控制台
            # if raise any exception, we assume the execution failed
            result_queue.put((None, False))

specification = r'''
from typing import List
import numpy as np

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
        completion_times[0, m] = completion_times[0, m-1] + processing_times[first_job, m]

    for i in range(1, num_jobs):
        job = schedule[i]
        completion_times[i, 0] = completion_times[i-1, 0] + processing_times[job, 0]
        for m in range(1, num_machines):
            completion_times[i, m] = max(completion_times[i, m-1], completion_times[i-1, m]) + processing_times[job, m]

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


'''


import os
import numpy as np
def load_datasets(dataset_folder):
    """
    Loads all datasets from the given folder, removing unnecessary job indices.

    :param dataset_folder: Path to the dataset folder.
    :return: Dictionary {dataset_name: job_matrix}
    """
    datasets = {}

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".txt"):  # Ensure only .txt files are read
                file_path = os.path.join(root, file)

                with open(file_path, "r") as f:
                    first_line = f.readline().strip().split()
                    n_jobs, n_machines = int(first_line[0]), int(first_line[1])  # Read job/machine count

                    raw_data = np.loadtxt(f)
                    jobs = raw_data[:, 1::2]  # Keep only even-indexed columns (machine times)

                    if jobs.shape != (n_jobs, n_machines):
                        print(f" Warning: Mismatch in expected dimensions for {file_path}, skipping dataset.")
                        continue  # Skip invalid dataset

                    datasets[file] = jobs  # Store dataset by filename

    return datasets



from implementation import funsearch
from implementation import config

# It should be noted that the if __name__ == '__main__' is required.
# Because the inner code uses multiprocess evaluation.
if __name__ == '__main__':

    data_path = "../data/"
    datasets = {}
    for subfolder in ["carlier", "heller", "reeves"]:
        datasets.update(load_datasets(os.path.join(data_path, subfolder)))

    print(f"Successfully loaded {len(datasets)} datasets.")

    instances = {
        "heller1.txt": datasets["heller1.txt"],
        "heller2.txt": datasets["heller2.txt"],
        "reeves1.txt": datasets["reeves1.txt"],
    }
    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=30)
    global_max_sample_num = 50  # if it is set to None, funsearch will execute an endless loop
    funsearch.main(
        specification=specification,
        inputs=instances,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        verbose=True,
        log_dir='../logs/evaluator.log'
    )
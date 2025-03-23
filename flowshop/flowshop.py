import os
import time
import json
import multiprocessing
import http.client

from typing import Collection, Any
from implementation import sampler

from prompts.prompts import *


api_key = 'Bearer sk-OmRJlpj2aI4A3GLvA4Bd841fCfB04b3e9eF6D0D9984f1719'

server_url = 'api.bltcy.ai'
max_tokens = 1024
model_name = 'chatgpt-4o-latest'

spec_file = 'flowshop_spec_priority.py'
add_prompt = base_prompt


def _trim_preface_of_body(sample: str) -> str:
    lines = sample.splitlines()
    func_body_lineno = 0
    find_def_declaration = False

    for lineno, line in enumerate(lines):
        if line[:3] == 'def':
            func_body_lineno = lineno
            find_def_declaration = True
            break

    if find_def_declaration:
        code = ''
        for line in lines[func_body_lineno+1:]:
            code += line + '\n'
        return code

    return sample


class LLMAPI(sampler.LLM):
    def __init__(
            self,
            samples_per_prompt: int,
            trim=True
    ):
        super().__init__(samples_per_prompt)
        self._additional_prompt = add_prompt
        self._trim = trim

    def draw_samples(
            self,
            prompt: str
    ) -> Collection[str]:
        return [
            self._draw_sample(prompt)
            for _ in range(self._samples_per_prompt)
        ]

    def _draw_sample(
            self,
            content: str
    ) -> str:
        prompt = '\n'.join([content, self._additional_prompt])

        while True:
            try:
                conn = http.client.HTTPSConnection(server_url)
                payload = json.dumps({
                    'max_tokens': max_tokens,
                    'model': model_name,
                    'messages': [
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ]
                })
                headers = {
                    'Authorization': api_key,
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }

                conn.request('POST', '/v1/chat/completions',
                             payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
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
    def __init__(
            self,
            verbose=False,
            numba_accelerate=False
    ):
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate

    def run(
            self,
            program:            str,
            function_to_run:    str,
            function_to_evolve: str,
            inputs:             Any,
            test_input:         str,
            timeout_seconds:    int,
            **kwargs
    ) -> tuple[Any, bool]:
        dataset = {
            test_input: inputs[test_input]
        }

        try:
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target = self._compile_and_run_function,
                args = (
                    program,
                    function_to_run,
                    function_to_evolve,
                    dataset,
                    self._numba_accelerate,
                    result_queue
                )
            )
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
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

    def _compile_and_run_function(
            self,
            program,
            function_to_run,
            function_to_evolve,
            dataset,
            numba_accelerate,
            result_queue
    ):
        try:
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program = program,
                    function_to_evolve = function_to_evolve
                )

            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            results = function_to_run(dataset)

            if not isinstance(results, (int, float)):
                result_queue.put((None, False))
                return

            result_queue.put((results, True))

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"[Sandbox Error]: {error_msg}")
            result_queue.put((None, False))


from flowshop_test.utils import load_datasets

from implementation import funsearch
from implementation import config


if __name__ == '__main__':
    data_path = '/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop/data'
    datasets = {}

    for subfolder in ['carlier', 'heller', 'reeves']:
        datasets.update(load_datasets(os.path.join(data_path, subfolder)))

    print(f'Successfully loaded {len(datasets)} datasets.')

    instances = {
        'carlier1.txt': datasets['carlier1.txt'],
        'carlier2.txt': datasets['carlier2.txt'],
        'heller1.txt': datasets['heller1.txt'],
        'heller2.txt': datasets['heller2.txt'],
        'reeves1.txt': datasets['reeves1.txt'],
        'reeves2.txt': datasets['reeves2.txt'],
        'reeves3.txt': datasets['reeves3.txt']
    }

    class_config = config.ClassConfig(llm_class=LLMAPI, sandbox_class=Sandbox)
    config = config.Config(samples_per_prompt=4, evaluate_timeout_seconds=30)
    global_max_sample_num = 50

    with open(spec_file, 'r') as f:
        specification = f.read()

    funsearch.main(
        specification=specification,
        inputs=instances,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        verbose=True,
        log_dir=f'./logs/evaluator_log/{spec_file.split('.')[0]}'
    )

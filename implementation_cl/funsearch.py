# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A single-threaded implementation of the FunSearch pipeline."""
from __future__ import annotations

# from collections.abc import Sequence

# RZ: there are multiple errors in the original code
# we should use typing.xxx rather than collections.abc.xxx
from typing import Any, Tuple, Sequence

from implementation_cl import code_manipulation
from implementation_cl import config as config_lib
from implementation_cl import stage_tracker


def _extract_function_names(specification: str) -> Tuple[str, str]:
    """Returns the name of the function to evolve and of the function to run.

    RZ: The so-called specification refers to the boilerplate code template for a task.
    The template MUST have two important functions decorated with '@funsearch.run', '@funsearch.evolve' respectively.
    The function labeled with '@funsearch.run' is going to evaluate the generated code (like fitness evaluation).
    The function labeled with '@funsearch.evolve' is the function to be searched (like 'greedy' in cap-set).
    This function (_extract_function_names) makes sure that these decorators appears in the specification.
    """
    run_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'run'))
    if len(run_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.run`.')
    evolve_functions = list(code_manipulation.yield_decorated(specification, 'funsearch', 'evolve'))
    if len(evolve_functions) != 1:
        raise ValueError('Expected 1 function decorated with `@funsearch.evolve`.')
    return evolve_functions[0], run_functions[0]


def _replace_with_evolved_function(
        file_path: str,
        function_to_evolve: str,
        function_to_run: str,
        evolved_function: code_manipulation.Function
):
    with open(file_path, 'r') as f:
        code = f.read()

    program = code_manipulation.text_to_program(code)
    for i, func in enumerate(program.functions):
        if func.name == function_to_evolve:
            program.functions[i] = evolved_function
            break
    else:
        raise ValueError(f'Function {function_to_evolve} not found in {file_path}')

    with open(file_path, 'w') as f:
        f.write(program.get_decorated_program_str(function_to_evolve, function_to_run))


def _file_backup(src: str):
    import shutil
    dst = src.split('.')[0] + '_copy.py'
    shutil.copy(src, dst)
    return dst


def main(
        spec_file_path: str,
        inputs: Sequence[Any],
        config: config_lib.Config,
        max_sample_nums_per_stage: int | None,
        max_stage_iter: int | None,
        class_config: config_lib.ClassConfig,
        **kwargs
):
    """Launches a FunSearch experiment.
    RZ:
    Args:
        specification: the boilerplate code for the problem.
        inputs       : the data instances for the problem (see 'bin_packing_utils.py').
        config       : config file.
        max_sample_nums: the maximum samples nums from LLM. 'None' refers to no stop.
    """

    spec_file_path = _file_backup(spec_file_path)

    cur_stage_idx = 0
    cur_iter = 0

    while cur_stage_idx < len(inputs) and cur_iter < max_stage_iter:

        cur_iter += 1

        with open(spec_file_path, 'r') as f:
            specification = f.read()

        function_to_evolve, function_to_run = _extract_function_names(specification)  # only function names
        template = code_manipulation.text_to_program(specification)  # entire python file as string

        st = stage_tracker.StageTracker(
            cur_stage=cur_stage_idx,
            max_sample_nums=max_sample_nums_per_stage,
            config=config,
            class_config=class_config
        )

        is_passed, evolved_func = st.main_process(
            function_to_evolve,
            function_to_run,
            template,
            inputs[cur_stage_idx],
            **kwargs
        )

        if is_passed:
            print(f"Passed Stage-{cur_stage_idx}, the evolved function is:")
            print(evolved_func)
            cur_stage_idx += 1

            _replace_with_evolved_function(
                spec_file_path, function_to_evolve, function_to_run, evolved_func
            )
        else:
            print(f"Failed to pass Stage-{cur_stage_idx}, retrying......")

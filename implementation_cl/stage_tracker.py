# Implemented by CGY
# Wrapper for the main process

from typing import Dict

from implementation_cl import evaluator
from implementation_cl import programs_database
from implementation_cl import sampler
from implementation_cl import profile
from implementation_cl import config as config_lib
from implementation_cl import code_manipulation

import os


class StageTracker:
    """
    Responsible for tracking each stage.

    CGY: Wrapping the main process of FunSearch to run continuously.
    """

    def __init__(
            self,
            cur_stage: int,
            max_sample_nums: int,
            config: config_lib.Config,
            class_config: config_lib.ClassConfig,
    ):
        self._cur_stage = cur_stage
        self._max_sample_nums = max_sample_nums
        self._config = config
        self._class_config = class_config

    def main_process(
            self,
            function_to_evolve: str,
            function_to_run: str,
            template: code_manipulation.Program,
            inputs: Dict,
            **kwargs
    ) -> tuple[bool, code_manipulation.Function]:
        """
        Main process of FunSearch.

        CGY: FunSearch main.

        Args:
            function_to_evolve:     the name of the evolve function
            function_to_run:        the name the run function
            template:               the specification
            inputs:                 input data for evaluation

        Returns:
            a tuple contains whether current run has passed
            the current stage and the final evolved function.
        """

        print(f"=====Stage {self._cur_stage} Started=====")

        database = programs_database.ProgramsDatabase(
            self._config.programs_database, template, function_to_evolve
        )

        # get log_dir and create profiler
        log_dir = kwargs.get('log_dir', None)
        if log_dir is None:
            profiler = None
        else:
            profiler = profile.Profiler(os.path.join(log_dir, f'Stage-{self._cur_stage}'))

        evaluators = [
            evaluator.Evaluator(
                database,
                template,
                function_to_evolve,
                function_to_run,
                inputs,
                timeout_seconds=self._config.evaluate_timeout_seconds,
                sandbox_class=self._class_config.sandbox_class,
            )
            for _ in range(self._config.num_evaluators)
        ]

        # We send the initial implementation to be analysed by one of the evaluators.
        initial = template.get_function(function_to_evolve).body
        baseline_score = evaluators[0].analyse(
            initial, island_id=None, version_generated=None, cur_stage=self._cur_stage, profiler=profiler
        )

        print(f"=====Stage-{self._cur_stage} Baseline Score: {baseline_score}=====")

        assert baseline_score is not None  # CGY: baseline score is compulsory

        sampler.Sampler.initialize_global_sample_nums()  # CGY: reset the class variable for the coming run

        # Set global max sample nums.
        samplers = [
            sampler.Sampler(
                database,
                evaluators,
                self._config.samples_per_prompt,
                max_sample_nums=self._max_sample_nums,
                llm_class=self._class_config.llm_class
            )
            for _ in range(self._config.num_samplers)
        ]

        for s in samplers:
            s.sample(profiler=profiler)

        print(f"=====Stage {self._cur_stage} Ended=====")

        # CGY: best score and function at current stage
        best_score = profiler.get_cur_best_score()
        best_func = profiler.get_cur_best_function()

        return best_score >= baseline_score, best_func

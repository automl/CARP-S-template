from __future__ import annotations

import ConfigSpace as CS
import numpy as np

from typing import Any
from carps.benchmarks.problem import Problem
from carps.loggers.abstract_logger import AbstractLogger
from carps.optimizers.optimizer import Optimizer
from carps.utils.trials import TrialInfo, TrialValue
from carps.utils.types import Incumbent
from carps.loggers.file_logger import FileLogger


# Inherit from carps' Optimizer class
class RandomOptimizer(Optimizer):
    def __init__(
        self,
        # These are required arguments
        problem: Problem,
        task: Task,
        loggers: list[AbstractLogger] | None = None,
        *,  # You can put whatever things you'd like for your optimizer here
        seed: int,
    ) -> None:
        # Query the task to know more.
        # TODO: We need docs on what is in a `Task`
        if task.is_multifidelity:
            raise NotImplementedError(
                "Multifidelity optimization is not supported yet."
            )

        super().__init__(problem, task, loggers)

        # You might need to convert the ConfigurationSpace object into one of your own...
        self.configspace: CS.ConfigurationSpace = problem.configspace
        self.search_space = self.convert_configspace(problem.configspace)

        self._seed = seed
        self._rng: np.random.Generator | None = np.random.default_rng(seed)
        self._history: dict[str, tuple[TrialInfo, TrialValue]] = {}
        self._trial_counter: int = 0

    # This is called just before the optimizer is `run()`
    def _setup_optimizer(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._trial_counter = 0

    # Checkout CARP's existing optimizers to learn more about how different optimizers
    # adapt the ConfigurationSpace
    def convert_configspace(self, configspace: CS.ConfigurationSpace) -> list[dict]:
        our_search_space = []
        for hp_name, hp in self.configspace.items():
            if isinstance(hp, CS.UniformFloatHyperparameter):
                hp = {
                    "name": hp_name,
                    "type": "float",
                    "bounds": (hp.lower, hp.upper),
                    "log": hp.log,
                }
            elif isinstance(hp, CS.UniformIntegerHyperparameter):
                hp = {
                    "name": hp_name,
                    "type": "int",
                    "bounds": (hp.lower, hp.upper),
                    "log": hp.log,
                }
            elif isinstance(hp, CS.CategoricalHyperparameter):
                if hp.weights is not None:  # Weight categorical
                    raise NotImplementedError()
                hp = {"name": hp_name, "type": "categorical", "values": hp.choices}
            else:
                # Please checkout the ConfigSpace documentation for
                # more hyperparameter types
                raise NotImplementedError(f"Unsupported hyperparameter type: {hp}")

            our_search_space.append(hp)

        return our_search_space

    def convert_to_trial(self, name: str, raw_config: dict[str, Any]) -> TrialInfo:
        # Now we need to convert it into a `TrialInfo`
        # This contains a ConfigSpace.Configuration, along with some more
        # meta-information. For the most part, you should create the
        # Configuration with some dictionary of values and be good to go!
        configuration = CS.Configuration(self.configspace, values=raw_config)

        return TrialInfo(
            name=name,
            config=configuration,
            budget=None,  # Needs to be specified if multi-fidelity
            seed=None,  # Seeded benchmarks
            instance=None,  # Benchmark/optimizers with multiple instances
        )

    def get_current_incumbent(self) -> Incumbent:
        return min(
            self._history.values(),
            key=lambda info_value: info_value[1].cost,
        )

    def ask(self) -> TrialInfo:
        assert self._rng is not None

        # Create a dictionary from our random sampling
        raw_config = {}
        for hp in self.search_space:
            name = hp["name"]
            _type = hp["type"]
            if _type == "float":
                lower, upper = hp["bounds"]
                raw_config[name] = self._rng.uniform(lower, upper)
            elif _type == "int":
                lower, upper = hp["bounds"]
                raw_config[name] = self._rng.integers(lower, upper)
            elif _type == "categorical":
                raw_config[name] = self._rng.choice(hp["values"])
            else:
                raise NotImplementedError(
                    f"Unsupported hyperparameter '{name}' with type: {_type}."
                )

        # It's good to give each configuration a name.
        # In this case, since we only support black-box problems, with no
        # specific extra-information (i.e. budget in the case of multi-fidelity),
        # then we can just use a simple counter.
        self._trial_counter += 1
        name = f"random-{self._trial_counter}"

        trial_info = self.convert_to_trial(name=name, raw_config=raw_config)
        return trial_info

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        assert trial_info.name is not None

        # We don't need to update any model but we hold on to what happened
        # for use with `get_current_incumbent()`
        self._history[trial_info.name] = (trial_info, trial_value)


if __name__ == "__main__":
    from carps.benchmarks.dummy_problem import DummyProblem
    from carps.utils.task import Task
    from rich import print as rich_print

    SEED = 42

    # Create a dummy problem
    problem = DummyProblem(return_value=42)
    task = Task(n_trials=10, n_objectives=len(problem.configspace))
    rich_print(task)

    # Create our optimizer and see that it runs
    opt = RandomOptimizer(problem=problem, task=task, seed=SEED)
    best_found = opt.run()
    rich_print(best_found)

    # By default, the framework will not save any intermediate data
    # However we can add a `FileLogger` to have CARP-S save the data
    # to disk, which can be retrieved later.
    # There is also a DatabaseLogger which you can find more about in the
    # documentation.
    overwrite = True
    problem = DummyProblem(
        return_value=42,
        loggers=[FileLogger(directory="output-dir", overwrite=overwrite)],
    )
    task = Task(n_trials=10, n_objectives=len(problem.configspace))
    opt = RandomOptimizer(problem=problem, task=task, seed=SEED)
    best_found = opt.run()

    # Right now, there will be raw logs of the run written to disk.
    # We can ask CARP-S to gather them all up and put them together for us!

    from carps.analysis.gather_data import filelogs_to_df

    df = filelogs_to_df("output-dir")

    # NOTE: This will read *all* the results for all experiments in the
    # specified directory when called! If you just want to load
    # the results, you can find them at `output-dir/logs.csv`.
    # Since this is just one small experiment, it will be cheap!
    #
    # df = pd.read_csv("output-dir/logs.csv")

    print(df)

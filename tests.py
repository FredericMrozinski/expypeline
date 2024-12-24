import unittest
import functools
from typing import List, Optional

import expypeline as expy


def general_exp_step(key_to_read: Optional[str], key_to_write: str,
                     val_to_expect, val_to_write, step_tracker: List[str], tester: unittest.TestCase, state: expy.ExpState):
    step_tracker.append(state.step_tag)
    if key_to_read is not None:
        tester.assertEqual(state[key_to_read], val_to_expect)
    state[key_to_write] = val_to_write
    tester.assertEqual(state[key_to_write], val_to_write)


class ExPypelineTests(unittest.TestCase):

    def test_single_step_experiment(self):
        """
        The goal of this test case is to assess whether the framework can run experiments that only consist
        of one step and are thus not passed as ExpPipelineLevelList but only as ExpStep.
        """
        suite = expy.ExpSuite(None)

        step_tracker = []

        step_fun = functools.partial(general_exp_step, None, "test_key", None, "test_val", step_tracker, self)
        step = expy.ExpStep("Single step", step_fun)

        suite.queue_experiment("Simple linear experiment", step)
        suite.run()

        self.assertListEqual(step_tracker, ["Single step"])


    def test_simple_linear_pipeline(self):
        """
        The goal of this test case is to see whether simple linear chaining of ExpSteps works, i.e., many experiment
        steps after each other without case branching. Further, we test if ExpState allows access to states from
        previous states without testing or using overriding.
        :return:
        """
        suite = expy.ExpSuite(None)
        pipeline = None

        step_tracker: List[str] = []

        depth = 20

        for i in range(depth):
            step_fun = functools.partial(general_exp_step, str(i - 1) if i > 0 else None,
                                                                            str(i),
                                                                            f"value {i - 1}",
                                                                            f"value {i}", step_tracker,
                                                                            self)
            step = expy.ExpStep(f"Step {i}", step_fun)
            if pipeline is None:
                pipeline = step
            else:
                pipeline = pipeline.then(step)

        suite.queue_experiment("Simple linear experiment", pipeline)
        suite.run()

        self.assertListEqual(step_tracker, [f"Step {i}" for i in range(depth)])


    def test_state_overriding_linear_pipeline(self):
        suite = expy.ExpSuite(None)
        pipeline = None

        step_tracker: List[str] = []

        depth = 20

        for i in range(depth):
            step_fun = functools.partial(general_exp_step, "key" if i > 0 else None,
                                         "key",
                                         f"value {i - 1}",
                                         f"value {i}", step_tracker,
                                         self)
            step = expy.ExpStep(f"Step {i}", step_fun)
            if pipeline is None:
                pipeline = step
            else:
                pipeline = pipeline.then(step)

        suite.queue_experiment("Overriding states linear experiment", pipeline)
        suite.run()

        self.assertListEqual(step_tracker, [f"Step {i}" for i in range(depth)])


    def test_simple_branching_pipeline(self):
        suite = expy.ExpSuite(None)

        step_tracker: List[str] = []

        depth = 20

        init_fun = functools.partial(general_exp_step, None,
                                         "key",
                                         None,
                                         "a",
                                         self)
        pipeline = expy.ExpStep("Init step", init_fun)

        for i in range(depth):
            step_fun = functools.partial(general_exp_step, "key",
                                         "key",
                                         "a",
                                         f"value {i}", step_tracker,
                                         self)
            step = expy.ExpStep(f"Step {i}", step_fun)
            pipeline = pipeline.branch(step)

        suite.queue_experiment("Simple branching experiment", pipeline)
        suite.run()

        self.assertListEqual(step_tracker, [f"Step {i}" for i in range(depth)])
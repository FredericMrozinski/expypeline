import json
import traceback
from datetime import datetime
from typing import Optional, Callable, List

class ExpSuite:
    pass

class ExpState:
    def __init__(self, step_tag: str, prev_state: Optional['ExpState'] = None):
        # TODO remove step tag, not needed no more
        self._step_tag = step_tag
        self._prev_state = prev_state
        self._attributes = {}

    def __getitem__(self, key):
        if key in self._attributes:
            return self._attributes[key]
        elif self._prev_state is not None:
            return self._prev_state[key]
        else:
            raise KeyError("ExpState object does not hold attribute '" + key + "'")

    def __setitem__(self, key, value):
        if self._prev_state is not None and self._prev_state.has_key(key):
            print(f"WARNING: Attribute '{key}' already exists from a previous step. Setting this value will create "
                  f"a step-local copy of the attribute.") # TODO Replace by logger warning
        self._attributes[key] = value

    def has_key(self, key):
        if key in self._attributes:
            return True
        elif self._prev_state is not None:
            return self._prev_state.has_key(key)
        else:
            return False

    def derive_level_down(self, next_step_tag: str):
        chained = ExpState(step_tag=next_step_tag, prev_state=self)
        return chained

    def to_json_dict(self) -> dict:
        return self._attributes

class ExpPipeline:
    def __init__(self, exp_step: 'ExpStep', parent: 'ExpPipeline'):
        self._parent = parent
        self.exp_step = exp_step
        self.last_state = None
        self.next_pipelines = []

    def run(self, state: ExpState | None, parent_experiment: 'Experiment'):
        # Implement exception handling such that the pipeline still runs other steps
        # Implement seed resetting for every step -> also log seed to state

        self.last_state = state

        pipeline_tag = (" " + parent_experiment.experiment_name + " => "
                        + self.get_preceding_step_tags() + " > " + self.exp_step.tag + " ")

        print("\n\n----" + pipeline_tag + "-" * (98 - len(pipeline_tag)))

        try:
            self.exp_step.run(state)
        except Exception as e:
            print("\n×××× ! Exception thrown ! " + "×" * 76)
            print("× Pruning the following steps:" + " " * 71 + "×")
            for tag in self.get_subsequent_step_tags():
                print("× " + tag + " " * max(0, 99 - len(tag)) + "×")
            print("×" + " " * 100 + "×")
            print("× What?:" + " " * 93 + "×")

            for line in traceback.format_exc().splitlines():
                print("× " + line + " " * max(0, 98 - len(line)) + " ×")

            print("×" * 102 + "\n")
        else:
            for next_pipeline in self.next_pipelines:
                next_pipeline.run(state.derive_level_down(next_pipeline.exp_step.tag), parent_experiment)

    def run_order_str(self,
                      run_order_str: str,
                      branch_levels: List[int],
                      current_level: int,
                      line_prefix: Optional[str] = "") -> str:
        def level_prefix(branch_levels: List[int], current_level: int) -> str:
            res = line_prefix
            for l in range(current_level):
                if l in branch_levels:
                    res += "| "
                else:
                    res += "  "
            return res + '|\n' + res

        run_order_str += "\n" + level_prefix(branch_levels, current_level)
        run_order_str += "+---" + self.exp_step.tag

        for i in range(len(self.next_pipelines)):
            branch_levels_cpy = branch_levels.copy()
            if len(self.next_pipelines) > 1 and i < len(self.next_pipelines) - 1:
                branch_levels_cpy.append(current_level + 1)
            run_order_str = self.next_pipelines[i].run_order_str(run_order_str, branch_levels_cpy, current_level + 1,
                                                                 line_prefix)

        return run_order_str

    def set_next_steps(self, *steps):
        for step in steps:
            self.next_pipelines.append(ExpPipeline(step, parent=self))

    def get_subsequent_step_tags(self, concat_preceding: bool = True) -> List[str]:
        if len(self.next_pipelines) == 0:
            return [self.exp_step.tag]
        else:
            step_tags = []
            own_tag = self.exp_step.tag
            if concat_preceding:
                own_tag = self.get_preceding_step_tags() + " > " + own_tag
            step_tags.append(own_tag)

            for next_pipeline in self.next_pipelines:
                tmp_tags = next_pipeline.get_subsequent_step_tags(concat_preceding=False)
                for i in range(len(tmp_tags)):
                    tmp_tags[i] = own_tag + " > " + tmp_tags[i]

                step_tags.extend(tmp_tags)
            return step_tags

    def get_preceding_step_tags(self) -> str:
        if self._parent is None:
            return ""
        else:
            parent_str = self._parent.get_preceding_step_tags()
            if parent_str != "":
                return parent_str + " > " + self._parent.exp_step.tag
            else:
                return self._parent.exp_step.tag

    def state_to_json_dict(self) -> dict:
        step_dict = {
            "_step_" : self.exp_step.tag,
            "_step_runtime_" : str(self.exp_step.timestamps["end"] - self.exp_step.timestamps["begin"]),
            "_step_state_" : self.last_state.to_json_dict(),
            "_subsequent_steps_" : [next_step.state_to_json_dict() for next_step in self.next_pipelines],
        }
        return step_dict

class ExpPipelineLevelList:
    def __init__(self, pipeline: Optional[ExpPipeline] = None):
        if pipeline is not None:
            self._pipelines_at_level = [pipeline]
        else:
            self._pipelines_at_level = []
        self._root = None

    def run_order_str(self, line_prefix: Optional[str] = ""):
        run_order_str = line_prefix + "EXPERIMENT PIPELINE:"

        root = self._root if self._root is not None else self

        for pipeline in root._pipelines_at_level:
            run_order_str = pipeline.run_order_str(run_order_str, [], 0, line_prefix)

        return run_order_str

    def then(self, *steps) -> 'ExpPipelineLevelList':
        next_level_list = ExpPipelineLevelList()

        if self._root is None:
            next_level_list._root = self
        else:
            next_level_list._root = self._root

        for pipeline in self._pipelines_at_level:
            pipeline.set_next_steps(*steps)
            next_level_list._pipelines_at_level.extend(pipeline.next_pipelines)

        return next_level_list

    def run(self, parent_experiment: 'Experiment'):
        if self._root is None:
            for pipeline in self._pipelines_at_level:
                pipeline.run(parent_experiment.root_exp_state.derive_level_down(pipeline.exp_step.tag),
                             parent_experiment)
        else:
            self._root.run(parent_experiment)

    def state_to_json_dict(self) -> dict:
        if self._root is None:
            step_dict = {
                "_steps_" : [next_step.state_to_json_dict() for next_step in self._pipelines_at_level]
            }
            return step_dict
        else:
            return self._root.state_to_json_dict()

class ExpStep:
    def __init__(self, tag: str, step: Callable[[ExpState], None]):
        self._step = step
        self.timestamps = {}
        self.tag = tag

    def run(self, state: ExpState):
        self.timestamps["begin"] = datetime.now()
        self._step(state)
        self.timestamps["end"] = datetime.now()

    def then(self, *next_steps) -> ExpPipelineLevelList: # TODO type annotate param
        pipeline_level_list = ExpPipelineLevelList(ExpPipeline(self, parent=None))
        return pipeline_level_list.then(*next_steps)

class Experiment:
    def __init__(self, experiment_name: str, experiment_pipeline: ExpPipelineLevelList | ExpStep):
        self.experiment_name = experiment_name
        if isinstance(experiment_pipeline, ExpStep):
            experiment_pipeline = ExpPipelineLevelList(ExpPipeline(experiment_pipeline, None))
        self.experiment_pipeline = experiment_pipeline
        self.timestamps = {}
        self.root_exp_state = ExpState(experiment_name, None)

    def run(self):
        print("== BEGINNING NEW EXPERIMENT " + "=" * 74)
        print("    NAME: " + self.experiment_name)
        print(self.experiment_pipeline.run_order_str("    "))

        self.timestamps["begin"] = datetime.now()
        self.experiment_pipeline.run(self)
        self.timestamps["end"] = datetime.now()

        print("== ENDING EXPERIMENT " + "=" * 81)

    def get_exp_state_str(self) -> str:
        exp_dict = {
            "_experiment_" : self.experiment_name,
            "_experiment_begin_" : self.timestamps["begin"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_end_" : self.timestamps["end"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_runtime_" : str(self.timestamps["end"] - self.timestamps["begin"]),
            **self.experiment_pipeline.state_to_json_dict()
        }
        return json.dumps(exp_dict, indent=4, sort_keys=True, default=lambda o: str(o))




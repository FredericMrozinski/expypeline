import json
import os
import socket
import copy
import traceback
import base64
from datetime import datetime
from typing import Optional, Callable, List

version = "0.2.0"

def path_safe(path: str):
    path = path.replace("\\", "/")
    if path.endswith("/"):
        return path[:-1]
    return path

def folder_safe(folder: str):
    return path_safe(folder) # TODO implement

class ExpSuite:
    def __init__(self, output_directory: Optional[str] = None):
        self.output_directory: Optional[str] = path_safe(output_directory) if output_directory else None
        self.experiments: List['Experiment'] = []
        self.current_experiment = None

    def queue_experiment(self, experiment_name: str, experiment_pipeline: 'ExpPipelineLevelList | ExpStep'):
        self.experiments.append(Experiment(experiment_name, experiment_pipeline, self))

    def get_shared_data_directory(self):
        return self.output_directory + "/shared"

    def get_experiment_specific_directory(self, experiment: 'Experiment'):
        return self.output_directory + "/" + folder_safe(experiment.experiment_name)

    def run(self):
        print("  ______      _____                  _ _\n"            
" |  ____|    |  __ \\                | (_)\n"           
" | |__  __  _| |__) |   _ _ __   ___| |_ _ __   ___\n" 
" |  __| \\ \\/ /  ___/ | | | '_ \\ / _ \\ | | '_ \\ / _ \\\n"
" | |____ >  <| |   | |_| | |_) |  __/ | | | | |  __/\n"
" |______/_/\\_\\_|    \\__, | .__/ \\___|_|_|_| |_|\\___|\n"
"                     __/ | |                        \n"
"                    |___/|_|                  v" + str(version) + "\n\n")

        print("QUEUED EXPERIMENTS:")
        for experiment in self.experiments:
            print(f"   - {experiment.experiment_name}")

        for experiment in self.experiments:
            experiment.run()

class ExpState:
    def __init__(self, prev_state: Optional['ExpState'] = None):
        self._prev_state = prev_state
        self._attributes = {}
        self.step_tag: str = None

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

    def derive_level_down(self):
        chained = ExpState(prev_state=self)
        return chained

    def to_json_dict(self) -> dict:
        return self._attributes


class ExpStepRunnable:
    def __init__(self, tag: str, step: Callable[[ExpState], None]):
        self._step = step
        self.logs = {}
        self.tag = tag
        self.last_state = None

    def run(self, state: ExpState, prev_step_tags: List[str], pruned: bool = False):
        self.last_state = state

        state.step_tag = self.tag
        context_step_tag = " > ".join([*prev_step_tags, self.tag])

        if pruned:
            self.logs["status"] = "pruned"
            print("×× PRUNED ×× -- " + context_step_tag + " " + "-" * (85 - len(context_step_tag)))
            return False

        print("\n\n-- " + context_step_tag + " " + "-" * (98 - len(context_step_tag)))

        self.logs["begin"] = datetime.now()

        try:
            self._step(state)
            self.logs["status"] = "success"
        except Exception as e:
            print("\n×××× ! Exception thrown ! " + "×" * 76)
            stacktrace = traceback.format_exc()
            for line in stacktrace.splitlines():
                print("× " + line + " " * max(0, 98 - len(line)) + " ×")
            print("×" * 102 + "\n")

            self.logs["status"] = "fail"
            self.logs["status_details"] = (f"Exception thrown: {repr(e)}. "
                                           f"Base64 encoded stacktrace: {base64.b64encode(stacktrace.encode("ascii")).decode("ascii")}")

        self.logs["end"] = datetime.now()
        if self.logs["status"] == "success":
            return True
        return False

    def to_json_dict(self) -> dict:
        return {
            "_name_": self.tag,
            "_meta_": self.logs,
            "_state_": {**self.last_state.to_json_dict()},
        }


class ExpPipelineRunnable:
    def __init__(self):
        self.heads: List[ExpStepRunnable] = []
        self.subsequents: List[ExpPipelineRunnable] = []

    def append_subsequent(self, subsequent_pipeline: 'ExpPipelineRunnable'):
        for subsequent in self.subsequents:
            subsequent.append_subsequent(copy.deepcopy(subsequent_pipeline))
        if len(self.subsequents) == 0:
            self.subsequents.append(copy.deepcopy(subsequent_pipeline))

    def get_order_str(self, line_prefix: str):
        return self._get_order_str_rec("", [], 0, line_prefix=line_prefix)

    def _get_order_str_rec(self,
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

        for i in range(len(self.subsequents)):
            run_order_str += "\n" + level_prefix(branch_levels, current_level)
            run_order_str += "+---" + self.heads[i].tag
            branch_levels_cpy = branch_levels.copy()
            if len(self.heads) > 1 and i < len(self.heads) - 1:
                branch_levels_cpy.append(current_level)
            if self.subsequents[i] is not None:
                run_order_str = self.subsequents[i]._get_order_str_rec(run_order_str, branch_levels_cpy,
                                                                     current_level + 1,
                                                                     line_prefix)

        return run_order_str

    def run(self, state: ExpState, prev_step_tags: List[str], pruned: bool = False):
        for head, subsequent in zip(self.heads, self.subsequents):
            current_state = state.derive_level_down()
            successful = head.run(current_state, prev_step_tags.copy(), pruned=pruned)
            if subsequent is not None:
                subsequent.run(current_state, [*prev_step_tags, head.tag], pruned=not successful or pruned)

    def to_json_dict(self) -> dict:
        json_dict = []
        for head, subsequent in zip(self.heads, self.subsequents):
            head_json_dict = head.to_json_dict()
            if subsequent is None:
                head_json_dict["_subsequent_"] = "None"
            else:
                head_json_dict["_subsequent_"] = subsequent.to_json_dict()
            json_dict.append(head_json_dict)

        return json_dict

class ExpPipelineBuilder:
    def __init__(self):
        self.root = self

    def then(self, subsequent: 'ExpPipelineBuilder'):
        pass

    def branch(self, subsequent: 'ExpPipelineBuilder'):
        pass

    def build(self) -> ExpPipelineRunnable:
        pass

    def _build_rec(self) -> ExpPipelineRunnable:
        pass

class ExpPipelineBuilderPiece(ExpPipelineBuilder):
    def __init__(self, step: 'ExpStepRunnable'):
        ExpPipelineBuilder.__init__(self)
        self.heads: List['ExpStepRunnable'] = []
        if step is not None:
            self.heads.append(step)
        self.subsequent: ExpPipelineBuilder = None

    def then(self, subsequent: 'ExpPipelineBuilder') -> ExpPipelineBuilder:
        if subsequent is None:
            raise PermissionError("Cannot call ExpPipeline.then(..) on a pipeline that already has a subsequent "
                                  "pipeline. If you want to branch off multiple subsequent steps at the same level,"
                                  "please call ExpPipeline.branch(..), instead.")

        subsequent = ExpPipelineBuilderPiece(subsequent.root)
        subsequent.root = self.root
        subsequent.heads[0].root = self.root
        self.subsequent = subsequent
        return subsequent

    def branch(self, subsequent: 'ExpPipelineBuilder') -> ExpPipelineBuilder:
        # Turn ExpStep into ExpPipelinePiece if necessary
        if isinstance(subsequent, ExpStep):
            subsequent = ExpPipelineBuilderPiece(subsequent)

        self.heads.append(subsequent.root)
        subsequent.root = self.root
        return self

    def build(self) -> ExpPipelineRunnable:
        return self.root._build_rec()

    def _build_rec(self) -> ExpPipelineRunnable:
        runnable = ExpPipelineRunnable()

        for head in self.heads:
            # If we have nested branches or nested straight pipelines, we want to unnest them
            built_head = head._build_rec()
            built_subsequent = self.subsequent._build_rec() if self.subsequent is not None else None
            if isinstance(head, ExpPipelineBuilderPiece):
                for i in range(len(built_head.heads)):
                    nested_head = built_head.heads[i]
                    runnable.heads.append(nested_head)
                    nested_subsequent = built_head.subsequents[i] if len(built_head.subsequents) > 0 else None
                    runnable.subsequents.append(nested_subsequent)
                    if built_subsequent is not None:
                        if runnable.subsequents[-1] is None:
                            runnable.subsequents[-1] = copy.deepcopy(built_subsequent)
                        else:
                            runnable.subsequents[-1].append_subsequent(built_subsequent)
            elif isinstance(head, ExpStep):
                # ExpSteps are wrapped into an ExpPipelineRunnable when built. Here, we unwrap them because we don't
                # want that wrapping in this case
                runnable.heads.append(built_head.heads[0])
                runnable.subsequents.append(built_subsequent)

        return runnable


class ExpStep(ExpPipelineBuilder):
    def __init__(self, tag: str, step: Callable[[ExpState], None]):
        ExpPipelineBuilder.__init__(self)
        self._step = step
        self.tag = tag

    def then(self, subsequent: 'ExpPipelineBuilder') -> ExpPipelineBuilder:
        res = ExpPipelineBuilderPiece(self)
        res = res.then(subsequent)
        return res

    def branch(self, subsequent: 'ExpPipelineBuilder') -> ExpPipelineBuilder:
        res = ExpPipelineBuilderPiece(self)
        res = res.branch(subsequent)
        return res

    def build(self) -> ExpPipelineRunnable:
        return self._build_rec()

    def _build_rec(self) -> ExpPipelineRunnable:
        runnable = ExpPipelineRunnable()
        step = ExpStepRunnable(self.tag, self._step)
        runnable.heads.append(step)
        return runnable


class Experiment:
    def __init__(self,
                 experiment_name: str,
                 experiment_pipeline: ExpPipelineBuilder,
                 experiment_suite: ExpSuite):
        self.experiment_name = experiment_name
        self.experiment_pipeline = experiment_pipeline.build()
        self.timestamps = {}
        self.root_exp_state = ExpState(None)
        self.experiment_suite = experiment_suite

    def run(self):
        print("\n\n== BEGINNING NEW EXPERIMENT " + "=" * 74)
        print("    NAME: " + self.experiment_name)
        print(self.experiment_pipeline.get_order_str("    "))

        self.timestamps["begin"] = datetime.now()
        self.experiment_pipeline.run(self.root_exp_state, [self.experiment_name], pruned=False)
        self.timestamps["end"] = datetime.now()

        print("\n-- ENDING & SAVING EXPERIMENT " + "-" * 72)
        state_str = self.get_exp_state_str()
        if self.experiment_suite.output_directory is not None and os.path.isdir(self.experiment_suite.output_directory):
            exp_out_path = self.experiment_suite.get_experiment_specific_directory(self)
            exp_out_path += "/run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists(exp_out_path):
                os.mkdir(exp_out_path)

            with open(exp_out_path + "/experiment_state.json", "w") as text_file:
                text_file.write(state_str)

        else:
            print("! No valid output path specified in ExpSuite ! Not writing experiment state but dumping here: ")
            print(state_str)
        print("== ENDED EXPERIMENT " + "=" * 82)

    def get_exp_state_str(self) -> str:
        exp_dict = {
            "_experiment_" : self.experiment_name,
            "_experiment_begin_" : self.timestamps["begin"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_end_" : self.timestamps["end"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_runtime_" : str(self.timestamps["end"] - self.timestamps["begin"]),
            "_host_system_" : socket.gethostname(),
            "_expy_version_" : version,
            "_steps_": self.experiment_pipeline.to_json_dict(),
        }
        return json.dumps(exp_dict, indent=4, sort_keys=True, default=lambda o: str(o))




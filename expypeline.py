# MIT License
#
# Copyright (c) 2024 Frederic Mrozinski
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import json
import os
import socket
import copy
import sys
import shutil
import time
import traceback
import logging
import base64
import platform, re, uuid
import hashlib
from datetime import datetime
from typing import Optional, Callable, List

logger = logging.getLogger("ExPypeline")
EXPYPELINE_LOG_LEVEL = 21

version = "0.5.1"

# TODO add Empty pipeline

# TODO replace by os functionality
def path_safe(path: str):
    path = path.replace("\\", "/")
    if path.endswith("/"):
        return path[:-1]
    return path

def folder_safe(folder: str):
    return path_safe(folder) # TODO implement

def _get_terminal_width():
    try:
        width = os.get_terminal_size()[0]
    except OSError:
        width = 80
    return width

class LogDecorationFilter(logging.Filter):
    def filter(self, record):
        # Add a level-based prefix (lowercase level name)
        if record.levelno != 21:
            record.msg = f"[{record.levelname}]{max(0, 8 - len(record.levelname)) * " "}{record.msg}"
        return True

class LogExportDecorationFilter(logging.Filter):
    def filter(self, record):
        to_return = ""
        while record.msg[0] == '\n':
            to_return += '\n'
            record.msg = record.msg[1:]
        record.msg = record.msg.replace('\n', '\n' + " " * 25)
        to_return += f"[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]    {record.msg}"
        record.msg = to_return
        return True

def _set_log_export_location(file_name):
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler.addFilter(LogExportDecorationFilter())
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    return file_handler

def _remove_log_export_handler(file_handler):
    file_handler.close()
    logging.getLogger().removeHandler(file_handler)

class ExpSuite:
    def __init__(self, output_directory: Optional[str] = None):
        self.output_directory: Optional[str] = path_safe(output_directory) if output_directory else None
        self.global_shared_directory =  os.path.join(self.output_directory, "shared/data")
        self.experiments: List['Experiment'] = []
        self.current_experiment = None
        self.reproducibility_handler: Optional[Callable[[float], None]] = None
        self.log_level_counter = LogLevelCounter()
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.getLogger().addHandler(self.log_level_counter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        prefix_filter = LogDecorationFilter()
        stream_handler.addFilter(prefix_filter)
        logger.addFilter(stream_handler)

        def step_name_based_seed_generator(level: int, state: 'ExpState') -> float:
            seed = 0
            for c in state.step_tag:
                seed += ord(c)
            return seed
        self.seed_generator: Callable[[int, 'ExpState'], float] = step_name_based_seed_generator


    def queue_experiment(self, experiment_name: str, experiment_pipeline: 'ExpPipelineLevelList | ExpStep'):
        self.experiments.append(Experiment(experiment_name, experiment_pipeline, self))

    def run(self, debug: bool = False):
        logger.log(EXPYPELINE_LOG_LEVEL, "  ______      _____                  _ _\n"            
" |  ____|    |  __ \\                | (_)\n"           
" | |__  __  _| |__) |   _ _ __   ___| |_ _ __   ___\n" 
" |  __| \\ \\/ /  ___/ | | | '_ \\ / _ \\ | | '_ \\ / _ \\\n"
" | |____ >  <| |   | |_| | |_) |  __/ | | | | |  __/\n"
" |______/_/\\_\\_|    \\__, | .__/ \\___|_|_|_| |_|\\___|\n"
"                     __/ | |                        \n"
"                    |___/|_|                  v" + str(version) + "\n\n")

        # Warn if no seed setter is passed
        if self.reproducibility_handler is None:
            logger.warning("No reproducibility handler has been set. Randomness will not be reproducible!")

        logger.log(EXPYPELINE_LOG_LEVEL, "QUEUED EXPERIMENTS:")
        for experiment in self.experiments:
            logger.log(EXPYPELINE_LOG_LEVEL, f"   - {experiment.experiment_name}")

        log_export_handler = None
        exp_summaries = []
        for experiment in self.experiments:
            if log_export_handler:
                _remove_log_export_handler(log_export_handler)

            root_exp_state = ExpState(None)
            root_exp_state._global_shared_data_dir = self.global_shared_directory
            root_exp_state._debug_mode = debug
            summary = experiment.run(root_exp_state)
            exp_summaries.append(summary)

            log_export_handler = experiment.log_export_handler

        run_file_location = self._write_exp_summaries(exp_summaries)
        return GlobalSummarizer(self.output_directory, run_file_location)

    def set_reproducibility_handler(self, reproducibility_handler:
            Optional[Callable[[float], None]]):
        self.reproducibility_handler = reproducibility_handler

    def _write_exp_summaries(self, summaries: List[dict]):
        runs_dir = os.path.join(self.output_directory, ".runs")
        if not os.path.isdir(runs_dir):
            os.makedirs(runs_dir)

        json_formatted = json.dumps(summaries, indent=4)
        summary_file_name = os.path.join(runs_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json")
        with open(summary_file_name, "w") as summary_file:
            summary_file.write(json_formatted)

        return summary_file_name

class ExpState:
    def __init__(self, prev_state: Optional['ExpState'] = None):
        self._prev_state = prev_state
        self._attributes = {}
        self.step_tag: str = None
        self.unique_step_id: str = None
        self._global_shared_data_dir = None
        self._run_shared_data_dir = None
        self._run_specific_data_dir = None
        self._experiment_suite: ExpSuite = None
        self._debug_mode = False
        self._logger = None

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

    # TODO add/override __contains__

    def has_key(self, key):
        if key in self._attributes:
            return True
        elif self._prev_state is not None:
            return self._prev_state.has_key(key)
        else:
            return False

    def _to_json_dict(self) -> dict:
        return self._attributes

    def _derive_level_down(self):
        chained = ExpState(prev_state=self)
        chained._global_shared_data_dir = self._global_shared_data_dir
        chained._run_shared_data_dir = self._run_shared_data_dir
        chained._run_specific_data_dir = self._run_specific_data_dir
        chained._experiment_suite = self._experiment_suite
        return chained

    def get_global_shared_data_dir(self):
        return self._global_shared_data_dir

    def get_run_shared_data_dir(self):
        return self._run_shared_data_dir

    def get_run_specific_data_dir(self):
        return self._run_specific_data_dir

    def get_exp_suite(self):
        return self._experiment_suite

    def is_debug_mode(self):
        return self._debug_mode

    def logger(self):
        if self._logger is None:
            self._logger = logging.getLogger(self.unique_step_id)
            self._logger.setLevel(logging.DEBUG)

            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            prefix_filter = LogDecorationFilter()
            stream_handler.addFilter(prefix_filter)
            self._logger.addFilter(stream_handler)

        return self._logger

    def debug(self, msg: str):
        self.logger().debug(msg)

    def info(self, msg: str):
        self.logger().info(msg)

    def warning(self, msg: str):
        self.logger().warning(msg)

    def error(self, msg: str):
        self.logger().error(msg)

    def fatal(self, msg: str):
        self.logger().fatal(msg)


class ExpStepRunnable:
    def __init__(self, tag: str, step: Callable[[ExpState], None]):
        self._step = step
        self.logs = {}
        self.tag = tag
        self.last_state = None

    def run(self, state: ExpState, prev_step_tags: List[str], run_status: dict, pruned: bool = False):
        self.last_state = state

        run_status["total"] += 1

        state.step_tag = self.tag
        # Make runtime dependent step-id
        id_base = self.tag + str(time.time_ns())
        unique_step_id = hashlib.md5(id_base.encode('utf-8')).hexdigest()[:12]
        self.logs["unique_step_id"] = unique_step_id
        state.unique_step_id = unique_step_id
        context_step_tag = " 🠞 ".join([*prev_step_tags, self.tag])

        if pruned:
            self.logs["status"] = "pruned"
            run_status["pruned"] += 1
            logger.warning("×× PRUNED ×× -- " + context_step_tag + " " + "-" * (85 - len(context_step_tag)))
            return False

        logger.log(EXPYPELINE_LOG_LEVEL, "\n── " + context_step_tag + " " + "─" * (_get_terminal_width() - len(context_step_tag) - 6))

        # (Re-)set seed
        seed = "undefined (no reproducibility handler set)"
        if state.get_exp_suite().reproducibility_handler is not None:
            seed = state.get_exp_suite().seed_generator(0, state)
            state.get_exp_suite().reproducibility_handler(seed)
        self.logs["random_seed"] = seed

        self.logs["begin"] = datetime.now()

        try:
            self._step(state)
            self.logs["status"] = "success"
            run_status["success"] += 1
        except (Exception, KeyboardInterrupt) as e:
            logger.log(EXPYPELINE_LOG_LEVEL, "")
            logger.error("")
            logger.log(EXPYPELINE_LOG_LEVEL, "×××× ! Exception thrown ! " + "×" * 76)
            stacktrace = traceback.format_exc()
            for line in stacktrace.splitlines():
                logger.log(EXPYPELINE_LOG_LEVEL, line)
            logger.log(EXPYPELINE_LOG_LEVEL, "×" * 102 + "\n")

            self.logs["status"] = "fail"
            self.logs["status_details"] = (f"Exception thrown: {repr(e)}. "
                                           f"Base64 encoded stacktrace: "
                                           f"{base64.b64encode(stacktrace.encode('ascii')).decode('ascii')}")
            run_status["error"] += 1

        self.logs["end"] = datetime.now()
        if self.logs["status"] == "success":
            return True
        return False

    def to_json_dict(self) -> dict:
        return {
            "_name_": self.tag,
            "_meta_": self.logs,
            "_state_": {**self.last_state._to_json_dict()},
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
        return f"{line_prefix}PIPELINE:\n" + self._get_order_str_rec("", [], 0, line_prefix=line_prefix)

    def _get_order_str_rec(self,
                      run_order_str: str,
                      branch_levels: List[int],
                      current_level: int,
                      line_prefix: Optional[str] = "") -> str:
        def level_prefix(branch_levels: List[int], current_level: int) -> str:
            res = line_prefix
            for l in range(current_level):
                if l in branch_levels:
                    res += "│   "
                else:
                    res += "    "
            return res #+ '│\n'

        for i in range(len(self.subsequents)):
            run_order_str += level_prefix(branch_levels, current_level)
            branch_prefix = "├─🠞 " if i < len(self.subsequents) - 1 else "└─🠞 "
            run_order_str += branch_prefix + self.heads[i].tag + "\n"
            branch_levels_cpy = branch_levels.copy()
            if len(self.heads) > 1 and i < len(self.heads) - 1:
                branch_levels_cpy.append(current_level)
            if self.subsequents[i] is not None:
                run_order_str = self.subsequents[i]._get_order_str_rec(run_order_str, branch_levels_cpy,
                                                                     current_level + 1,
                                                                     line_prefix)

        return run_order_str

    def run(self, state: ExpState, prev_step_tags: List[str], run_status: dict, pruned: bool = False):
        for head, subsequent in zip(self.heads, self.subsequents):
            current_state = state._derive_level_down()
            successful = head.run(current_state, prev_step_tags.copy(), run_status, pruned=pruned)
            if subsequent is not None:
                subsequent.run(current_state, [*prev_step_tags, head.tag], run_status, pruned=not successful or pruned)

    def to_json_dict(self) -> dict:
        json_dict = []
        for head, subsequent in zip(self.heads, self.subsequents):
            head_json_dict = head.to_json_dict()
            if subsequent is None:
                head_json_dict["_subsequent_"] = []
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
        # TODO ensure that a subsequent (also if nested into another pipeline) does not have a head with the
        # same name as already given

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


class Empty(ExpPipelineBuilder):
    def then(self, subsequent: 'ExpPipelineBuilder'):
        res = ExpPipelineBuilderPiece(subsequent)
        return res

    def branch(self, subsequent: 'ExpPipelineBuilder'):
        return self.then(subsequent)

    def build(self) -> ExpPipelineRunnable:
        raise RuntimeError("An Empty ExPypeline module cannot stand alone! Call then(..) or branch(..) on it, instead.")

    def _build_rec(self) -> ExpPipelineRunnable:
        raise RuntimeError("An Empty ExPypeline module cannot stand alone! Call then(..) or branch(..) on it, instead.")


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
        built = self._build_rec()
        return built

    def _build_rec(self) -> ExpPipelineRunnable:
        runnable = ExpPipelineRunnable()
        step = ExpStepRunnable(self.tag, self._step)
        runnable.heads.append(step)
        runnable.subsequents.append(None)
        return runnable


class Experiment:
    def __init__(self,
                 experiment_name: str,
                 experiment_pipeline: ExpPipelineBuilder,
                 experiment_suite: ExpSuite):
        self.experiment_name = experiment_name
        self.experiment_pipeline = experiment_pipeline.build()
        self.timestamps = {}
        self.experiment_suite = experiment_suite
        self.last_root_exp_state = None
        self.log_export_handler = None

    def run(self, root_exp_state: ExpState):
        self.last_root_exp_state = root_exp_state
        self.experiment_suite.log_level_counter.new_experiment()

        run_state_dict = {
            "total": 0,
            "success": 0,
            "pruned": 0,
            "error": 0,
        }

        # TODO replace all by os.join
        if self.experiment_suite.output_directory is not None:
            exp_out_path = os.path.join(self.experiment_suite.output_directory, path_safe(self.experiment_name))
            root_exp_state._run_shared_data_dir = os.path.join(exp_out_path, "shared/data")
            if root_exp_state.is_debug_mode():
                exp_out_path += "/debug"
                if os.path.exists(exp_out_path):
                    shutil.rmtree(exp_out_path)
            else:
                exp_out_path += "/run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            root_exp_state._run_specific_data_dir = exp_out_path + "/data"

            if not os.path.exists(root_exp_state.get_global_shared_data_dir()):
                os.makedirs(root_exp_state.get_global_shared_data_dir())
            if not os.path.exists(exp_out_path):
                os.makedirs(exp_out_path)
            if not os.path.exists(root_exp_state.get_run_shared_data_dir()):
                os.makedirs(root_exp_state.get_run_shared_data_dir())
            if not os.path.exists(root_exp_state.get_run_specific_data_dir()):
                os.makedirs(root_exp_state.get_run_specific_data_dir())

            log_export_location = os.path.join(exp_out_path, "log.txt")
            self.log_export_handler = _set_log_export_location(log_export_location)

        logger.log(EXPYPELINE_LOG_LEVEL, "\n\n\u2550\u2500 BEGINNING NEW EXPERIMENT \u2500" + "\u2550" * (_get_terminal_width() - 29))
        logger.log(EXPYPELINE_LOG_LEVEL, "    NAME: " + self.experiment_name)
        logger.log(EXPYPELINE_LOG_LEVEL, self.experiment_pipeline.get_order_str("    "))

        root_exp_state._experiment_suite = self.experiment_suite

        self.timestamps["begin"] = datetime.now()
        self.experiment_pipeline.run(root_exp_state, [self.experiment_name], run_state_dict, pruned=False)
        self.timestamps["end"] = datetime.now()

        logger.log(EXPYPELINE_LOG_LEVEL, "\n── ENDING & SAVING EXPERIMENT " + "─" * (_get_terminal_width() - 29))
        state_str = self.get_exp_state_str()
        if self.experiment_suite.output_directory is not None and os.path.isdir(self.experiment_suite.output_directory):
            with open(exp_out_path + "/experiment_state.json", "w") as text_file:
                text_file.write(state_str)

        else:
            logger.warning("! No valid output path specified in ExpSuite ! Not writing experiment state but dumping here: ")
            logger.log(EXPYPELINE_LOG_LEVEL, state_str)

        logger.log(EXPYPELINE_LOG_LEVEL, "\n\u2554\u2500 SUMMARY \u2500"
                   + "\u2550" * (_get_terminal_width() - 12) + "\u2557")


        exp_summary = {
            "Experiment": self.experiment_name,
            "Directory": os.path.relpath(exp_out_path, self.experiment_suite.output_directory),
            "Total steps": run_state_dict["total"],
            "Successful steps": run_state_dict["success"],
            "Crashed steps": run_state_dict["error"],
            "Pruned steps": run_state_dict["pruned"],
            "Warnings": self.experiment_suite.log_level_counter.last_experiment_log_level_counts['WARNING'],
            "Errors": self.experiment_suite.log_level_counter.last_experiment_log_level_counts['ERROR'],
            "Fatals": self.experiment_suite.log_level_counter.last_experiment_log_level_counts['FATAL'],
        }

        for summary_key, summary_val in exp_summary.items():
            logger.log(EXPYPELINE_LOG_LEVEL, f"\u2551  {summary_key}"
                       + (18 - len(summary_key)) * " " + str(summary_val)
                       + (_get_terminal_width() - len(str(summary_val)) - 21) * " " + "\u2551")
        logger.log(EXPYPELINE_LOG_LEVEL, "\u255A\u2500 ENDED EXPERIMENT \u2500"
                   + "\u2550" * (_get_terminal_width() - 21) + "\u255D")

        return exp_summary

    def get_exp_state_str(self) -> str:
        exp_dict = {
            "_experiment_" : self.experiment_name,
            "_experiment_begin_" : self.timestamps["begin"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_end_" : self.timestamps["end"].isoformat(sep=' ', timespec='milliseconds'),
            "_experiment_runtime_" : str(self.timestamps["end"] - self.timestamps["begin"]),
            "_debug_mode_" : self.last_root_exp_state.is_debug_mode(),
            "_system_" : {
                'platform': platform.system(),
                'platform-release': platform.release(),
                'platform-version': platform.version(),
                'architecture': platform.machine(),
                'hostname': socket.gethostname(),
                'ip-address': socket.gethostbyname(socket.gethostname()),
                'mac-address': ':'.join(re.findall('..', '%012x' % uuid.getnode())),
                'processor': platform.processor(),
                'python_version': sys.version,
            },
            "_expy_version_" : version,
            "_steps_": self.experiment_pipeline.to_json_dict(),
        }

        def safe_default_serializer(obj):
            try:
                return str(obj)
            except TypeError:
                return "UnserializableObject"

        return json.dumps(exp_dict, indent=4, sort_keys=True, default=safe_default_serializer)


class Summarizer:
    '''
    The goal of this class is to provide an interface for the final experiment state output (JSON)
    in a more readable format which the user might find more convenient to query.
    Why do we not save experiment states in this format by default?
        - We only save ExpState differences between ExpSteps. A user, however, will likely want to query
        a value as accessible at a certain ExpStep, independently of whether it was set in a previous step
   '''
    def __init__(self, parent_summarizer: 'Summarizer'):
        self.sub_dict = None
        self.parent_summarizer = None

class ParentLookupWrapper:
    def __init__(self, wrapped_dict: dict, parent_lookup_wrapper: 'ParentLookupWrapper'):
        self.wrapped_dict = wrapped_dict
        self.parent_lookup_wrapper = parent_lookup_wrapper

    def __getitem__(self, item):
        if item in self.wrapped_dict:
            return self.wrapped_dict[item]
        elif self.parent_lookup_wrapper is not None:
            return self.parent_lookup_wrapper[item]
        else:
            return None

    def __contains__(self, item):
        if item in self.wrapped_dict:
            return True
        elif self.parent_lookup_wrapper is not None:
            return item in self.parent_lookup_wrapper
        return False


class GlobalSummarizer(Summarizer):
    def __init__(self, top_level_expy_path: str, run_file_location: str):
        self.top_level_expy_path = top_level_expy_path
        self.run_file_location = run_file_location
        self.exp_summaries = {}
        self.summaries_loaded = False

    def _load_summaries(self):
        with open(os.path.join(self.top_level_expy_path, self.run_file_location)) as f:
            summaries = json.load(f)

        for summary in summaries:
            exp_name = summary["Experiment"]
            exp_location = os.path.join(summary["Directory"], "experiment_state.json")
            exp_dict = None
            with open(os.path.join(self.top_level_expy_path, exp_location)) as f:
                exp_dict = json.load(f)
            self.exp_summaries[exp_name] = ExperimentSummarizer(exp_dict, None)

    def __getitem__(self, item):
        if not self.summaries_loaded:
            self._load_summaries()
            self.summaries_loaded = True

        return self.exp_summaries[item]

class ExperimentSummarizer(Summarizer):
    def __init__(self, exp_dict: dict, parent_summarizer: Summarizer):
        super(ExperimentSummarizer, self).__init__(parent_summarizer)
        self.state = {}
        self.exp_dict = exp_dict
        self.steps: dict = {}
        for step in exp_dict["_steps_"]:
            self.steps[step["_name_"]] = ExpStepSummarizer(step, self)

    def __getitem__(self, item):
        res = None

        if item in self.steps:
            res = self.steps[item]
        if item in self.state:
            if res is not None:
                raise KeyError(f"Ambiguous item retrieval! The key {item} describes multiple values. Please "
                               f"specify the data retrieval by retrieving from 'steps', 'meta', or 'state'.")
            res = self.meta[item]
        if item in self.exp_dict:
            if res is not None:
                raise KeyError(f"Ambiguous item retrieval! The key {item} describes multiple values. Please "
                               f"specify the data retrieval by retrieving from 'steps', 'meta', or 'state'.")
            res = self.state[item]

        return res


class ExpStepSummarizer(Summarizer):
    def __init__(self, step_dict: dict, parent_summarizer: Summarizer):
        super(ExpStepSummarizer, self).__init__(parent_summarizer)
        self.state = ParentLookupWrapper(step_dict["_state_"], parent_summarizer.state)
        self.meta = step_dict["_meta_"]
        self.subsequents: dict = {}
        for subsequent in step_dict["_subsequent_"]:
            self.subsequents[subsequent["_name_"]] = ExpStepSummarizer(subsequent, self)

    def __getitem__(self, item):
        res = None

        if item in self.subsequents:
            res = self.subsequents[item]
        if item in self.meta:
            if res is not None:
                raise KeyError(f"Ambiguous item retrieval! The key {item} describes multiple values. Please "
                      f"specify the data retrieval by retrieving from 'subsequents', 'meta', or 'state'.")
            res = self.meta[item]
        if item in self.state:
            if res is not None:
                raise KeyError(f"Ambiguous item retrieval! The key {item} describes multiple values. Please "
                      f"specify the data retrieval by retrieving from 'subsequents', 'meta', or 'state'.")
            res = self.state[item]

        return res


class LogLevelCounter(logging.Handler):
    log_level_counts = None

    def __init__(self, *args, **kwargs):
        super(LogLevelCounter, self).__init__(*args, **kwargs)
        self.last_experiment_log_level_counts = None
        self.log_level_counts = {
            "WARNING": 0,
            "ERROR": 0,
            "FATAL": 0,
        }

    def new_experiment(self):
        self.last_experiment_log_level_counts = {
            "WARNING": 0,
            "ERROR": 0,
            "FATAL": 0,
        }

    def emit(self, record):
        l = record.levelname
        if l not in self.log_level_counts:
            self.log_level_counts[l] = 0
        self.log_level_counts[l] += 1
        if self.last_experiment_log_level_counts:
            if l not in self.last_experiment_log_level_counts:
                self.last_experiment_log_level_counts[l] = 0
            self.last_experiment_log_level_counts[l] += 1

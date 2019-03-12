import os, sys, re, shutil
import subprocess
import json
from math import ceil
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.benchmarking.benchmark import Benchmark
from autoPyTorch.utils.benchmarking.benchmark_pipeline import ForInstance, ForAutoNetConfig, ForRun, CreateAutoNet, SetAutoNetConfig
from autoPyTorch.data_management.data_manager import DataManager, ProblemType

import argparse

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--partial_benchmark", default=None, help="Only run a part of the benchmark. Run other parts later. 3-tuple: instance_slice, autonet_config_slice, run_number_range.")
    parser.add_argument("--time_bonus", default=[7200, 8200, 10800], type=int, nargs="+", help="Give the job some more time.")
    parser.add_argument("--memory_bonus", default=1000, type=int, help="Give the job some more memory. Unit: MB.")
    parser.add_argument("--result_dir", default=None, help="The dir to save the results")
    parser.add_argument("--output_dir", default=None, help="The dir to save the outputs")
    parser.add_argument("--template_args", default=[], nargs="+", type=str, help="Additional args specified in template")
    parser.add_argument("runscript", help="The script template used to submit job on cluster.")
    parser.add_argument('benchmark', help='The benchmark to run')
    args = parser.parse_args()

    # parse the runscript template
    with open(args.runscript, "r") as f:
        runscript_template = list(f)
    runscript_name = os.path.basename(args.runscript if not args.runscript.endswith(".template") else args.runscript[:-9])
    autonet_home = ConfigFileParser.get_autonet_home()
    host_config_orig = [l[13:] for l in runscript_template if l.startswith("#HOST_CONFIG ")][0].strip()
    host_config_file = os.path.join(autonet_home, host_config_orig) if not os.path.isabs(host_config_orig) else host_config_orig

    # parse template args
    runscript_template_args = [l[19:].strip().split() for l in runscript_template if l.startswith("#TEMPLATE_ARGUMENT ")]
    parsed_template_args = dict()
    for variable_name, default in runscript_template_args:
        try:
            value = [a.split("=")[1] for a in args.template_args if a.split("=")[0] == variable_name][0]
        except IndexError:
            value = default
        parsed_template_args[variable_name] = value
    
    # get benchmark config
    benchmark_config_file = args.benchmark

    benchmark = Benchmark()
    config_parser = benchmark.get_benchmark_config_file_parser()

    benchmark_config = config_parser.read(benchmark_config_file)
    benchmark_config.update(config_parser.read(host_config_file))
    config_parser.set_defaults(benchmark_config)

    # get ranges of runs, autonet_configs and instances
    all_configs = ForAutoNetConfig.get_config_files(benchmark_config, parse_slice=False)
    all_instances = ForInstance.get_instances(benchmark_config, instances_must_exist=True)

    runs_range = list(range(benchmark_config["num_runs"]))
    configs_range = list(range(len(all_configs)))
    instances_range = list(range(len(all_instances)))

    if (args.partial_benchmark is not None):
        split = args.partial_benchmark.split(',')
        if (len(split) > 0):
            instances_range = instances_range[ForInstance.parse_slice(split[0])]
        if (len(split) > 1):
            configs_range = configs_range[ForAutoNetConfig.parse_slice(split[1])]
        if (len(split) > 2):
            runs_range = list(ForRun.parse_range(split[2], benchmark_config["num_runs"]))
    
    # set up dict used used to make replacements in runscript
    base_dir = os.getcwd()
    result_dir = os.path.abspath(args.result_dir) if args.result_dir is not None else benchmark_config["result_dir"]
    outputs_folder = os.path.abspath(args.output_dir) if args.output_dir is not None else os.path.join(base_dir, "outputs")
    benchmark = args.benchmark if os.path.isabs(args.benchmark) else os.path.join(base_dir, args.benchmark)
    output_base_dir = os.path.join(outputs_folder, os.path.basename(benchmark).split(".")[0])
    replacement_dict = {
        "BASE_DIR": base_dir,
        "OUTPUTS_FOLDER": outputs_folder,
        "OUTPUT_BASE_DIR": output_base_dir,
        "AUTONET_HOME": autonet_home,
        "BENCHMARK": benchmark,
        "BENCHMARK_NAME": os.path.basename(benchmark).split(".")[0],
        "HOST_CONFIG": host_config_file,
        "ORIG_HOST_CONFIG": host_config_orig,
        "ORIG_BENCHMARK": args.benchmark,
        "RESULT_DIR": result_dir
    }
    replacement_dict.update(parsed_template_args)

    # create directories
    if not os.path.exists(outputs_folder):
        os.mkdir(outputs_folder)
    if not os.path.exists(output_base_dir):
        os.mkdir(output_base_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # iterate over all runs
    for run_number in runs_range:
        replacement_dict["RUN_NUMBER"] = run_number

        for config_id in configs_range:
            replacement_dict["CONFIG_ID"] = config_id
            replacement_dict["CONFIG_FILE"] = all_configs[config_id]

            # get autonet config
            dm = DataManager()
            dm.problem_type = {
                "feature_classification": ProblemType.FeatureClassification,
                "feature_multilabel": ProblemType.FeatureMultilabel,
                "feature_regression": ProblemType.FeatureRegression
            }[benchmark_config["problem_type"]]
            autonet = CreateAutoNet().fit(benchmark_config, dm)["autonet"]
            autonet_config_file = benchmark_config["autonet_configs"][config_id]
            SetAutoNetConfig().fit(benchmark_config, autonet, autonet_config_file, dm)
            autonet_config = autonet.get_current_autonet_config()

            # add autonet config specific stuff to replacement dict
            replacement_dict["NUM_WORKERS"] = autonet_config["min_workers"]
            replacement_dict["NUM_NODES"] = autonet_config["min_workers"] + (0 if autonet_config["run_worker_on_master_node"] else 1)
            replacement_dict["MEMORY_LIMIT_MB"] = autonet_config["memory_limit_mb"] + args.memory_bonus
            time_limit_base = autonet_config["max_runtime"] if autonet_config["max_runtime"] < float("inf") else (benchmark_config["time_limit"] - max(args.time_bonus))
            replacement_dict.update({("TIME_LIMIT[%s]" % i): int(t + time_limit_base) for i, t in enumerate(args.time_bonus)})
            replacement_dict["NUM_PROCESSES"] = max(autonet_config["torch_num_threads"], int(ceil(replacement_dict["MEMORY_LIMIT_MB"] / benchmark_config["memory_per_core"])))

            for instance_id in instances_range:
                replacement_dict["INSTANCE_ID"] = instance_id
                replacement_dict["INSTANCE_FILE"] = all_instances[instance_id]

                # create output subdirectory used fot this run
                output_dir = os.path.join(output_base_dir, "output_%s_%s_%s" % (instance_id, config_id, run_number))
                if os.path.exists(output_dir) and input("%s exists. Delete? (y/n)" %output_dir).startswith("y"):
                    shutil.rmtree(output_dir)
                os.mkdir(output_dir)
                replacement_dict["OUTPUT_DIR"] = output_dir

                # make replacements in runscript and get command to submit the job
                pattern = re.compile("|".join(map(lambda x: re.escape("$${" + x + "}"), replacement_dict.keys())))
                runscript = [pattern.sub(lambda x: str(replacement_dict[x.group()[3:-1]]), l) for l in runscript_template]
                command = [l[9:] for l in runscript if l.startswith("#COMMAND ")][0].strip()

                # save runscript
                with open(os.path.join(output_dir, runscript_name), "w") as f:
                    f.writelines(runscript)

                # submit job
                os.chdir(output_dir)
                print("Calling %s in %s" % (command, os.getcwd()))
                try:
                    command_output = subprocess.check_output(command, shell=True)
                except subprocess.CalledProcessError as e:
                    print("Warning: %s" % e)
                    command_output = str(e).encode("utf-8")
                    if not input("Continue (y/n)? ").startswith("y"):
                        raise
                os.chdir(base_dir)

                # save output and info data
                with open(os.path.join(output_dir, "call.info"), "w") as f:
                    print(command, file=f)
                    json.dump(replacement_dict, f)
                    print("", file=f)
                with open(os.path.join(output_dir, "call.info"), "ba") as f:
                    f.write(command_output)

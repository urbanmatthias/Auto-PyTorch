

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.benchmarking.benchmark import Benchmark

import argparse

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--run_id_range", default=None, help="An id for the run. A range of run ids can be given: start-stop.")
    parser.add_argument("--partial_benchmark", default=None, nargs="+", help="Only run a part of the benchmark. Run other parts later or in parallel. 3-tuple: instance_slice, autonet_config_slice, run_number_range.")
    parser.add_argument("--result_dir", default=None, help="Override result dir in benchmark config.")
    parser.add_argument("--host_config", default=None, help="Override some configs according to host specifics.")
    parser.add_argument("--plot_logs", default=[], nargs="+", help="List of metrics to plot. If not given, plot metric given in autonet config.")
    parser.add_argument("--only_finished_runs", action="store_true", help="Skip run folders, that do not contain a summary.json")
    parser.add_argument("--output_folder", default=None, help="Store the plots as pdf. Specify an output folder.")
    parser.add_argument("--scale_uncertainty", default=1, type=float, help="Scale the uncertainty")
    parser.add_argument("--agglomeration", default="mean", help="Choose between mean and median.")
    parser.add_argument("--font_size", default=12, type=int, help="Set font size.")
    parser.add_argument("--prefixes", default=[""], type=str, nargs="+", help="The prefixes to plot. Choices: none, train, val, test, ensemble, ensemble_test")
    parser.add_argument("--additional_trajectories", default=[], type=str, nargs="+", help="Path to json file describing additional trajectories")
    parser.add_argument('benchmark', help='The benchmark to visualize')

    args = parser.parse_args()

    run_id_range = args.run_id_range
    if args.run_id_range is not None:
        if "-" in args.run_id_range:
            run_id_range = range(int(args.run_id_range.split("-")[0]), int(args.run_id_range.split("-")[1]) + 1)
        else:
            run_id_range = range(int(args.run_id_range), int(args.run_id_range) + 1)
    
    benchmark_config_file = args.benchmark
    host_config_file = args.host_config

    benchmark = Benchmark()
    config_parser = benchmark.get_benchmark_config_file_parser()

    benchmark_config = config_parser.read(benchmark_config_file)
    benchmark_config.update(config_parser.read(host_config_file))

    if (args.result_dir is not None):
        benchmark_config['result_dir'] = os.path.abspath(args.result_dir)
    
    if (args.partial_benchmark is not None):
        if (len(args.partial_benchmark) > 0):
            benchmark_config['instance_slice'] = args.partial_benchmark[0]
        if (len(args.partial_benchmark) > 1):
            benchmark_config['autonet_config_slice'] = args.partial_benchmark[1]
        if (len(args.partial_benchmark) > 2):
            benchmark_config['run_number_range'] = args.partial_benchmark[2]

    benchmark_config['run_id_range'] = run_id_range
    benchmark_config['plot_logs'] = args.plot_logs
    benchmark_config['only_finished_runs'] = args.only_finished_runs
    benchmark_config['output_folder'] = args.output_folder
    benchmark_config['scale_uncertainty'] = args.scale_uncertainty
    benchmark_config['agglomeration'] = args.agglomeration
    benchmark_config['font_size'] = args.font_size
    benchmark_config['prefixes'] = [p if p != "none" else "" for p in args.prefixes]
    benchmark_config['additional_trajectories'] = args.additional_trajectories
    benchmark_config['benchmark_name'] = os.path.basename(args.benchmark).split(".")[0]
    
    benchmark.visualize_benchmark(**benchmark_config)

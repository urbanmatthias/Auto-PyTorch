

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.metalearning.meta_model_builder import MetaModelBuilder
from autoPyTorch.utils.benchmarking.benchmark import Benchmark
from hpbandster.core.nameserver import nic_name_to_host

import argparse
import json

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--run_id_range", default=None, help="An id for the run. A range of run ids can be given: start-stop.")
    parser.add_argument("--result_dir", default=None, help="Override result dir in benchmark config.")
    parser.add_argument("--only_finished_runs", action='store_true', help="Only consider finished runs")
    parser.add_argument('benchmark', help='The benchmark to evaluate from')

    args = parser.parse_args()

    run_id_range = args.run_id_range
    if args.run_id_range is not None:
        if "-" in args.run_id_range:
            run_id_range = range(int(args.run_id_range.split("-")[0]), int(args.run_id_range.split("-")[1]) + 1)
        else:
            run_id_range = range(int(args.run_id_range), int(args.run_id_range) + 1)
    
    config_file = args.benchmark

    benchmark = Benchmark()
    config_parser = benchmark.get_benchmark_config_file_parser()

    config = config_parser.read(config_file)

    if (args.result_dir is not None):
        config['result_dir'] = os.path.join(ConfigFileParser.get_autonet_home(), args.result_dir)

    config['run_id_range'] = run_id_range
    config["only_finished_runs"] = args.only_finished_runs
    config['benchmark_name'] = os.path.basename(args.benchmark).split(".")[0]
    config["metalearning_evaluate"] = True

    builder = MetaModelBuilder()
    builder.run(**config)
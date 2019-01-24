

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from autoPyTorch.utils.config.config_file_parser import ConfigFileParser
from autoPyTorch.utils.metalearning.meta_model_builder import MetaModelBuilder
from autoPyTorch.utils.benchmarking.benchmark import Benchmark
from hpbandster.core.nameserver import nic_name_to_host

import argparse

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for autonet.')
    parser.add_argument("--run_id_range", default=None, help="An id for the run. A range of run ids can be given: start-stop.")
    parser.add_argument("--result_dir", default=None, help="Override result dir in benchmark config.")
    parser.add_argument("--only_finished_runs", action='store_true', help="Only consider finished runs")
    parser.add_argument("--save_path", default=".", help="Store the meta learning models in given filename")
    parser.add_argument("--learn_warmstarted_model", action='store_true', help="Learn a warmstarted model")
    parser.add_argument("--learn_initial_design", action='store_true', help="Learn an initial_design")
    parser.add_argument("--calculate_loss_matrix_entry", default=-1, type=int, help="Calculate an entry of the cost matrix used for initial design")
    parser.add_argument("--loss_matrix_path", default="./loss_matrix.txt", type=str, help="Path to file where losses for initial design will be stored/are stored.")
    parser.add_argument("--memory_limit_mb", default=None, type=int)
    parser.add_argument("--time_limit_per_entry", default=None, type=int)
    parser.add_argument("--initial_design_max_total_budget", default=None, type=float)    
    parser.add_argument("--initial_design_convergence_threshold", default=None, type=float)
    parser.add_argument("--lock_dir", default=None, type=str)
    parser.add_argument('benchmark', help='The benchmark to learn from')

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
    config['save_path'] = args.save_path
    config['memory_limit_mb'] = args.memory_limit_mb
    config['time_limit_per_entry'] = args.time_limit_per_entry
    config["learn_warmstarted_model"] = args.learn_warmstarted_model
    config["learn_initial_design"] = args.learn_initial_design
    config["calculate_loss_matrix_entry"] = args.calculate_loss_matrix_entry 
    config["loss_matrix_path"] =  args.loss_matrix_path
    config["only_finished_runs"] = args.only_finished_runs
    config["initial_design_max_total_budget"] = args.initial_design_max_total_budget
    config["initial_design_convergence_threshold"] = args.initial_design_convergence_threshold
    config["lock_dir"] = args.lock_dir

    builder = MetaModelBuilder()
    builder.run(**config)
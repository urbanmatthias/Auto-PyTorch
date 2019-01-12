

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
    parser.add_argument("--save_path", default=".", help="Store the meta learning models in given filename")
    parser.add_argument("--num_processes", default=0, type=int, help="Number of available processes")
    parser.add_argument("--calculate_exact_incumbent_scores", action="store_true", help="Number of available processes")
    parser.add_argument("--network_interface_name", default="lo")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--distributed_id", default=0)
    parser.add_argument("--distributed_node", default=1)
    parser.add_argument("--distributed_dir", default=".")
    parser.add_argument("--memory_limit_mb", default=None, type=int)
    parser.add_argument("--time_limit_per_entry", default=None, type=int)
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
    config['num_processes'] = args.num_processes
    config['calculate_exact_incumbent_scores'] = args.calculate_exact_incumbent_scores
    config['distributed'] = args.distributed
    config['distributed_id'] = args.distributed_id
    config['distributed_dir'] = args.distributed_dir
    config['host'] = nic_name_to_host(args.network_interface_name)
    config['master'] = args.distributed_node == "1"
    config['memory_limit_mb'] = args.memory_limit_mb
    config['time_limit_per_entry'] = args.time_limit_per_entry


    builder = MetaModelBuilder()
    builder.run(**config)
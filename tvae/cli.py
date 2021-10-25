import sys
import argparse
from tvae.experiments import (
    ffa_modeling_fc6,
    ffa_modeling_fc6_functional,
    ffa_modeling_tdann_functional,
    ffa_modeling_pretrained_alexnet,
    ffa_modeling_tdann,
    ffa_modeling_tdann_functional
)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'tvae.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()

if __name__ == "__main__":
    main()
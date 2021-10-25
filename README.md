# Modeling Category-Selective Cortical Regions with Topographic Variational Autoencoders

## Getting Started
#### Install requirements with Anaconda:
`conda env create -f environment.yml`

#### Activate the conda environment
`conda activate tvae`

#### Install the tvae package
Install the tvae package inside of your conda environment. This allows you to run experiments with the `tvae` command. At the root of the project directory run (using your environment's pip):
`pip3 install -e .`

If you need help finding your environment's pip, try `which python`, which should point you to a directory such as `.../anaconda3/envs/tvae/bin/` where it will be located.

#### (Optional) Setup Weights & Biases:
This repository uses Weight & Biases for experiment tracking. By deafult this is set to off. However, if you would like to use this (highly recommended!) functionality, all you have to do is set `'wandb_on': True` in the experiment config, and set your account's project and entity names in the `tvae/utils/logging.py` file.

For more information on making a Weight & Biases account see [(creating a weights and biases account)](https://app.wandb.ai/login?signup=true) and the associated [quickstart guide](https://docs.wandb.com/quickstart).

## Running an experiment
To evaluate the selectivity of pretrained alexnet (the non-topographic baseline), you can run:
- `tvae --name 'ffa_modeling_pretrained_alexnet'`

To train and evaluate the selectivity of the TVAE for objects, faces, bodies, and places, you can run:
- `tvae --name 'ffa_modeling_fc6'`

To train and evaluate the selectivity of the the TDANN for objects, faces, bodies, and places, you can run:
- `tvae --name 'ffa_modeling_tdann'`

To evaluate the selectivity of the TVAE on abstract catagories (animacy vs. inanimacy): 
- `tvae --name 'ffa_modeling_fc6_functional'`

To evaluate the selectivity of the TDANN on abstract catagories (animacy vs. inanimacy): 
- `tvae --name 'ffa_modeling_tdann_functional'`

These 'functional' experiment files can also be easily modified to test selectivity to big vs. small objects by simply changing the directories of the input images.

## Basics of the framework
- All experiments can be found in `tvae/experiments/`, and begin with the model specification, followed by the experiment config. 

#### Model Architecutre Options
- `'mu_init'`: *int*, Initalization value for mu parameter
- `'s_dim'`: *int*, Dimensionality of the latent space
- `'k'`: *int*, size of the summation kernel used to define the local topographic structure
- `'group_kernel'`: *tuple of int*, defines the size of the kernel used by the grouper, exact definition and relationship to W varies for each experiment.

#### Training Options
- `'wandb_on'`: *bool*, if True, use weights & biases logging
- `'lr'`: *float*, learning rate
- `'momentum'`: *float*, standard momentum used in SGD
- `'max_epochs'`: *int*, total training epochs
- `'eval_epochs'`: *int*, epochs between evaluation on the test (for MNIST)
- `'batch_size'`: *int*, number of samples per batch
- `'n_is_samples'`: *int*, number of importance samples when computing the log-likelihood on MNIST.

## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.
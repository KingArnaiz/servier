# Servier Data Science Technical Test

This repository contains message passing neural networks for molecular property prediction as described in the paper [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and as used in the paper [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1) for molecules.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Installing from PyPi](#option-1-installing-from-pypi)
  * [Option 2: Installing from source](#option-2-installing-from-source)
  * [Docker](#docker)
- [Web Interface](#web-interface)
- [Within Python](#within-python)
- [Data](#data)
- [Training](#training)
  * [Train/Validation/Test Splits](#trainvalidationtest-splits)
  * [Loss functions](#loss-functions)
  * [Metrics](#metrics)
  * [Cross validation and ensembling](#cross-validation-and-ensembling)
  * [Aggregation](#aggregation)
  * [Additional Features](#additional-features)
    * [Custom Features](#molecule-level-custom-features)
    * [RDKit 2D Features](#molecule-level-rdkit-2d-features)
    * [Atomic Features](#atom-level-features)
  * [Pretraining](#pretraining)
  * [Missing target values](#missing-target-values)
  * [Weighted training by target and data](#weighted-training-by-target-and-data)
  * [Caching](#caching)
- [Predicting](#predicting)
  * [Uncertainty Estimation](#uncertainty-estimation)
  * [Uncertainty Calibration](#uncertainty-calibration)
  * [Uncertainty Evaluation Metrics](#uncertainty-evaluation-metrics)
- [Hyperparameter Optimization](#hyperparameter-optimization)
  * [Choosing the Search Parameters](#choosing-the-search-parameters)
  * [Checkpoints and Parallel Operation](#checkpoints-and-parallel-operation)
  * [Random or Directed Search](#random-or-directed-search)
  * [Manual Trials](#manual-trials)
- [Interpreting Model Prediction](#interpreting)
- [TensorBoard](#tensorboard)

## Requirements
To use `servier` with GPUs, you will need:
 * cuda >= 8.0
 * cuDNN

## Installation

Servier can either be installed from PyPi via pip or from source (i.e., directly from this git repo). The PyPi version includes a vast majority of Servier functionality, but some functionality is only accessible when installed from source.

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

Then proceed to either option below to complete the installation. Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/).

### Option 1: Installing from PyPi

1. `conda create -n servier python=3.8`
2. `conda activate servier`
3. `conda install -c conda-forge rdkit`
4. `pip install git+https://github.com/bp-kelley/descriptastorus`
5. `pip install servier`

### Option 2: Installing from source

1. `git clone https://github.com/servier/servier.git`
2. `cd servier`
3. `conda env create -f environment.yml`
4. `conda activate servier`
5. `pip install -e .`

### Docker

Servier can also be installed with Docker. Docker makes it possible to isolate the Servier code and environment. To install and run our code in a Docker container, follow these steps:

1. `git clone https://github.com/servier/servier.git`
2. `cd servier`
3. Install Docker from [https://docs.docker.com/install/](https://docs.docker.com/install/)
4. `docker build -t servier .`
5. `docker run -it servier:latest`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs.
Alternatively, with Docker 19.03+, you can specify the `--gpus` command line option instead.

In addition, you will also need to ensure that the CUDA toolkit version in the Docker image is compatible with the CUDA driver on your host machine.
Newer CUDA driver versions are backward-compatible with older CUDA toolkit versions.
To set a specific CUDA toolkit version, add `cudatoolkit=X.Y` to `environment.yml` before building the Docker image.

## Web Interface
You can start the web interface on your local machine in two ways. Flask is used for development mode while gunicorn is used for production mode.

### Flask

Run `servier_web` (or optionally `python web.py` if installed from source) and then navigate to [localhost:5000](http://localhost:5000) in a web browser.

### Gunicorn

Gunicorn is only available for a UNIX environment, meaning it will not work on Windows. It is not installed by default with the rest of servier, so first run:

```
pip install gunicorn
```

Next, navigate to `servier/web` and run `gunicorn --bind {host}:{port} 'wsgi:build_app()'`. This will start the site in production mode.
   * To run this server in the background, add the `--daemon` flag.
   * Arguments including `init_db` and `demo` can be passed with this pattern: `'wsgi:build_app(init_db=True, demo=True)'` 
   * Gunicorn documentation can be found [here](http://docs.gunicorn.org/en/stable/index.html).

## Data

In order to train a model, you must provide training data containing molecules (as SMILES strings) and known target values. Data sets are available in the `data` directory.

Servier can either train on a single target ("single tasking") or on multiple targets simultaneously ("multi-tasking").

There are four current supported dataset types. Targets with unknown values can be left as blanks.
* **Regression.** Targets are float values. With bounded loss functions or metrics, the values may also be simple inequalities (e.g., >7.5 or <5.0).
* **Classification.** Targets are binary (i.e. 0s and 1s) indicators of the classification.
* **Multiclass.** Targets are integers (starting with zero) indicating which class the datapoint belongs to, out of a total number of exclusive classes indicated with `--multiclass_num_classes <int>`.

The data file must be be a **CSV file with a header row**. 

By default, it is assumed that the SMILES are in the first column (can be changed using `--number_of_molecules`) and the targets are in the remaining columns. However, the specific columns containing the SMILES and targets can be specified using the `--smiles_columns <column_1> ...` and `--target_columns <column_1> <column_2> ...` flags, respectively.

## Training

To train a model, run:
```
servier_train --data_path <path> --dataset_type <type> --save_dir <dir>
```
where `<path>` is the path to a CSV file containing a dataset, `<type>` is one of [classification, regression, multiclass, spectra] depending on the type of the dataset, and `<dir>` is the directory where model checkpoints will be saved.

For example:
```
servier_train --data_path data/sample_train.csv --dataset_type classification --save_dir model1_checkpoints
```

A full list of available command-line arguments can be found in [servier/args.py](https://github.com/servier/servier/blob/master/servier/args.py).

If installed from source, `servier_train` can be replaced with `python train.py --data_path data/sample_train.csv --dataset_type classification --save_dir model1_checkpoints`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with `--metric <metric>`.
* `--save_dir` may be left out if you don't want to save model checkpoints.
* `--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the `save_dir`.

### Train/Validation/Test Splits

Our code supports several methods of splitting data into train, validation, and test sets.

* **Random.** By default, the data will be split randomly into train, validation, and test sets.
* **Scaffold.** Alternatively, the data can be split by molecular scaffold so that the same scaffold never appears in more than one split. This can be specified by adding `--split_type scaffold_balanced`.
* **Random With Repeated SMILES.** Some datasets have multiple entries with the same SMILES. To constrain splitting so the repeated SMILES are in the same split, use the argument `--split_type random_with_repeated_smiles`.
* **Separate val/test.** If you have separate data files you would like to use as the validation or test set, you can specify them with `--separate_val_path <val_path>` and/or `--separate_test_path <test_path>`. If both are provided, then the data specified by `--data_path` is used entirely as the training data. If only one separate path is provided, the `--data_path` data is split between train data and either val or test data, whichever is not provided separately.

When data contains multiple molecules per datapoint, scaffold and repeated SMILES splitting will only constrain splitting based on one of the molecules. The key molecule can be chosen with the argument `--split_key_molecule <int>`, with the default setting using an index of 0 indicating the first molecule.

By default, both random and scaffold split the data into 80% train, 10% validation, and 10% test. This can be changed with `--split_sizes <train_frac> <val_frac> <test_frac>`. The default setting is `--split_sizes 0.8 0.1 0.1`. If a separate validation set or test set is provided, the split defaults to 80%-20%. Splitting involves a random component and can be seeded with `--seed <seed>`. The default setting is `--seed 0`.

### Loss functions

The loss functions available for training are dependent on the selected dataset type. Loss functions other than the defaults can be selected from the supported options with the argument `--loss_function <function>`.
* **Regression.** mse (default), bounded_mse, mve (mean-variance estimation, a.k.a. heteroscedastic loss), evidential.
* **Classification.** binary_cross_entropy (default), mcc (a soft version of Matthews Correlation Coefficient), dirichlet (a.k.a. evidential classification)
* **Multiclass.** cross_entropy (default), mcc (a soft version of Matthews Correlation Coefficient)

The regression loss functions `mve` and `evidential` function by minimizing the negative log likelihood of a predicted uncertainty distribution. If used during training, the uncertainty predictions from these loss functions can be used for uncertainty prediction during prediction tasks.
### Metrics

Metrics are used to evaluate the success of the model against the test set as the final model score and to determine the optimal epoch to save the model at based on the validation set. The primary metric used for both purposes is selected with the argument `--metric <metric>` and additional metrics for test set score only can be added with `--extra_metrics <metric1> <metric2> ...`. Supported metrics are dependent on the dataset type. Unlike loss functions, metrics do not have to be differentiable.
* **Regression.** rmse (default), mae, mse, r2, bounded_rmse, bounded_mae, bounded_mse (default if bounded_mse is loss function).
* **Classification.** auc (default), prc-auc, accuracy, binary_cross_entropy, f1, mcc.
* **Multiclass.** cross_entropy (default), accuracy, f1, mcc.

When a multitask model is used, the metric score used for evaluation at each epoch or for choosing the best set of hyperparameters during hyperparameter search is obtained by taking the mean of the metric scores for each task. Some metrics scale with the magnitude of the targets (most regression metrics), so geometric mean instead of arithmetic mean is used in those cases in order to avoid having the mean score dominated by changes in the larger magnitude task.

### Cross validation and ensembling

k-fold cross-validation can be run by specifying `--num_folds <k>`. The default is `--num_folds 1`. Each trained model will have different data splits. The reported test score will be the average of the metrics from each fold.

To train an ensemble, specify the number of models in the ensemble with `--ensemble_size <n>`. The default is `--ensemble_size 1`. Each trained model within the ensemble will share data splits. The reported test score for one ensemble is the metric applied to the averaged prediction across the models. Ensembling and cros-validation can be used at the same time.

### Additional Features

While the model works very well on its own, especially after hyperparameter optimization, we have seen that additional features can further improve performance on certain datasets. The additional features can be added at the atom-, bond, or molecule-level. Molecule-level features can be either automatically generated by RDKit or custom features provided by the user.

#### Molecule-Level Custom Features

If you install from source, you can modify the code to load custom features as follows:

1. **Generate features:** If you want to generate features in code, you can write a custom features generator function in `servier/features/features_generators.py`. Scroll down to the bottom of that file to see a features generator code template.
2. **Load features:** If you have features saved as a numpy `.npy` file or as a `.csv` file, you can load the features by using `--features_path /path/to/features`. Note that the features must be in the same order as the SMILES strings in your data file. Also note that `.csv` files must have a header row and the features should be comma-separated with one line per molecule. By default, provided features will be normalized unless the flag `--no_features_scaling` is used.

#### Molecule-Level RDKit 2D Features

As a starting point, we recommend using pre-normalized RDKit features by using the `--features_generator rdkit_2d_normalized --no_features_scaling` flags. In general, we recommend NOT using the `--no_features_scaling` flag (i.e. allow the code to automatically perform feature scaling), but in the case of `rdkit_2d_normalized`, those features have been pre-normalized and don't require further scaling.

### Pretraining

Pretraining can be carried out using previously trained checkpoint files to set some or all of the initial values of a model for training. Additionally, some model parameters from the previous model can be frozen in place, so that they will not be updated during training.

Parameters from existing models can be used for parameter-initialization of a new model by providing a checkpoint of the existing model using either
 * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training of the old model). This will walk the directory, and load all `.pt` files it finds.
 * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
 * `--checkpoint_paths <list of paths>` A list of paths to multiple model checkpoint (`.pt`) files.
when training the new model. The model architecture of the new model should resemble the architecture of the old model - otherwise some or all parameters might not be loaded correctly. If any of these options are specified during training, any argument provided with `--ensemble_size` will be overwritten and the ensemble size will be specified as the number of checkpoint files that were provided, with each submodel in the ensemble using a separate checkpoint file for initialization. When using these options, new model parameters are initialized using the old checkpoint files but all parameters remain trainable (no frozen layers from these arguments).

Certain portions of the model can be loaded from a previous model and frozen so that they will not be trainable, using the various frozen layer parameters. A path to a checkpoint file for frozen parameters is provided with the argument `--checkpoint_frzn <path>`. If this path is provided, the parameters in the MPNN portion of the model will be specified from the path and frozen. Layers in the FFNN portion of the model can also be applied and frozen in addition to freezing the MPNN using `--frzn_ffn_layers <number-of-layers>`. Model architecture of the new model should match the old model in any layers that are being frozen, but non-frozen layers can be different without affecting the frozen layers (e.g., MPNN alone is frozen and new model has a larger number of FFNN layers). Parameters provided with `--checkpoint_frzn` will overwrite initialization parameters from `--checkpoint_path` (or similar) that are frozen in the new model. At present, only one checkpoint can be provided for the `--checkpoint_frzn` and those parameters will be used for any number of submodels if `--ensemble_size` is specified. If multiple molecules (with multiple MPNNs) are being trained in the new model, the default behavior is for both of the new MPNNs to be frozen and drawn from the checkpoint. Only the first MPNN will be frozen and subsequent MPNNs still allowed to train if `--freeze_first_only` is specified.

### Weighted Training by Target and Data

By default, each task in multitask training and each provided datapoint are weighted equally for training. Weights can be specified in either case to allow some tasks in training or some specified data points to be weighted more heavily than others in the training of the model.

Using the `--target_weights` argument followed by a list of numbers equal in length to the number of tasks in multitask training, different tasks can be given more weight in parameter updates during training. For instance, in a multitask training with two tasks, the argument `--target_weights 1 2` would give the second task twice as much weight in model parameter updates. Provided weights must be non-negative. Values are normalized to make the average weight equal 1. Target weights are not used with the validation set for the determination of early stopping or in evaluation of the test set.

Using the `--data_weights_path` argument followed by a path to a data file containing weights will allow each individual datapoint in the training data to be given different weight in parameter updates. Formatting of this file is similar to provided features CSV files: they should contain only a single column with one header row and a numerical value in each row that corresponds to the order of datapoints provided with `--data_path`. Data weights should not be provided for validation or test sets if they are provided through the arguments `--separate_test_path` or `--separate_val_path`. Provided weights must be non-negative. Values are normalized to make the average weight equal 1. Data weights are not used with the validation set for the determination of early stopping or in evaluation of the test set.

## Predicting

To load a trained model and make predictions, run `predict.py` and specify:
* `--test_path <path>` Path to the data to predict on.
* A checkpoint by using either:
  * `--checkpoint_dir <dir>` Directory where the model checkpoint(s) are saved (i.e. `--save_dir` during training). This will walk the directory, load all `.pt` files it finds, and treat the models as an ensemble.
  * `--checkpoint_path <path>` Path to a model checkpoint file (`.pt` file).
* `--preds_path` Path where a CSV file containing the predictions will be saved.

For example:
```
servier_predict --test_path data/sample_test.csv --checkpoint_dir model2_checkpoints --preds_path sample_test_preds.csv
```
or
```
servier_predict --test_path data/sample_test.csv --checkpoint_path model2_checkpoints/fold_0/model_0/model.pt --preds_path sample_test_preds.csv
```

Predictions made on an ensemble of models will return the average of the individual model predictions. To return the individual model predictions as well, include the `--individual_ensemble_predictions` argument.

If installed from source, `servier_predict` can be replaced with `python predict.py --test_path data/sample_test.csv --checkpoint_dir model2_checkpoints --preds_path sample_test_preds.csv` or `python predict.py --test_path data/sample_test.csv --checkpoint_path model2_checkpoints/fold_0/model_0/model.pt --preds_path sample_test_preds.csv`.

## Hyperparameter Optimization

Although the default message passing architecture works well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to improvement in performance. We have automated hyperparameter optimization via Bayesian optimization (using the [hyperopt](https://github.com/hyperopt/hyperopt) package). The default hyperparameter optimization will search for the best configuration of hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:
```
servier_hyperopt --data_path <data_path> --dataset_type <type> --num_iters <int> --config_save_path <config_path>
```
where `<int>` is the number of hyperparameter trial configurations to try and `<config_path>` is the path to a `.json` file where the optimal hyperparameters will be saved. If installed from source, `servier_hyperopt` can be replaced with `python hyperparameter_optimization.py`. Additional training arguments can also be supplied during submission, and they will be applied to all included training iterations (`--epochs`, `--aggregation`, `--num_folds`, `--gpu`, `--ensemble_size`, `--seed`, etc.). The argument `--log_dir <dir_path>` can optionally be provided to set a location for the hyperparameter optimization log.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:
```
servier_train --data_path <data_path> --dataset_type <type> --config_path <config_path>
```

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run hyperparameter_optimization.py separately on each split's training and validation data with test held out.

### Choosing the Search Parameters

The parameter space being searched can be changed to include different sets of model hyperparameters. These can be selected using the argument `--search_parameter_keywords <list-of-keywords>`. The available keywords are listed below. Some keywords refer to bundles of parameters or other special behavior. Note that the search ranges for each parameter is hardcoded and can be viewed or changed in `servier/hyperopt_utils.py`.

Special keywords
* basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, and linked_hidden_size.
* linked_hidden_size - search for hidden_size and ffn_hidden_size, but constrained for them to have the same value. This allows search through both but with one fewer degree of freedom.
* learning_rate - search for max_lr, init_lr, final_lr, and warmup_epochs.
* all - include search for all inidividual keyword options
Individual supported parameters
* activation, aggregation, aggregation_norm, batch_size, depth, dropout, ffn_hidden_size, ffn_num_layers, final_lr, hidden_size, init_lr, max_lr, warmup_epochs

Choosing to include additional search parameters should be undertaken carefully. The number of possible parameter combinations increases combinatorially with the addition of more hyperparameters, so the search for an optimal configuration will become more difficult accordingly. The recommendation from Hyperopt is to use at least 10 trials per hyperparameter for an appropriate search as a rule of thumb, but even more will be necessary at higher levels of search complexity or to obtain better convergence to the optimal hyperparameters. Steps to reduce the complexity of a search space should be considered, such as excluding low-sensitivity parameters or those for which a judgement can be made ahead of time. Splitting the search into two steps can also reduce overall complexity. The `all` search option should only be used in situations where the dataset is small and a very large number of trials can be used.

For best results, the `--epochs` specified during hyperparameter search should be the same as in the intended final application of the model. Learning rate parameters are especially sensitive to the number of epochs used. Note that the number of epochs is not a hyperparameter search option.

The search space for init_lr and final_lr values are defined as fractions of the max_lr value. The search space for warmup_epochs is set by fraction of the `--epochs` training argument. The search for aggregation_norm values is only relevant when the aggregation function is set to norm and can otherwise be neglected. If a separate training argument is provided that is included in the search parameters, the search will overwrite the specified value (e.g., `--depth 5 --search_parameter_keywords depth`).

## Interpreting

It is often helpful to provide explanation of model prediction (i.e., this molecule is toxic because of this substructure). Given a trained model, you can interpret the model prediction using the following command:
```
servier_interpret --data_path data/sample_test.csv --checkpoint_dir model1_checkpoints/fold_0/ --property_id 1
```

If installed from source, `servier_interpret` can be replaced with `python interpret.py`.

## TensorBoard

During training, TensorBoard logs are automatically saved to the same directory as the model checkpoints. To view TensorBoard logs, first install TensorFlow with `pip install tensorflow`. Then run `tensorboard --logdir=<dir>` where `<dir>` is the path to the checkpoint directory. Then navigate to [http://localhost:5000](http://localhost:5000).

## Results
We run the three models with 5 number of folds and a combination of ensemble learning. We have the folllowing results:

Results on classification test sets (AUC score, the higher the better)

### Model 1
| Folds |	AUC |
| :---: | :---: |
| 1 |  0.75 |
| 2 |  0.77|
| 3 |  0.75 |
| 4 | 0.79  |
| 5 |  0.70 |
| Overall |  0.75 ± 0.03 |

### Model 2
| Folds |	AUC |
| :---: | :---: |
| 1 |  0.74 |
| 2 |  0.75|
| 3 |  0.71 |
| 4 | 0.77  |
| 5 |  0.65 |
| Overall |  0.73 ± 0.04 |

### Model 3
| Folds |	AUC |
| :---: | :---: |
| 1 |  0.69 |
| 2 |  0.72|
| 3 |  0.72 |
| 4 | 0.72  |
| 5 |  0.69 |
| Overall |  0.71 ± 0.01 |

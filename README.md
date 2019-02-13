# disentanglement_lib
![Sample visualization](https://github.com/google-research/disentanglement_lib/blob/master/sample.gif?raw=true)

**disentanglement_lib** is an open-source library for research on learning disentangled representation.
It supports a variety of different models, metrics and data sets:

* *Models*: BetaVAE, FactorVAE, BetaTCVAE, DIP-VAE
* *Metrics*: BetaVAE score, FactorVAE score, Mutual Information Gap, SAP score, DCI, MCE
* *Data sets*: dSprites, Color/Noisy/Scream-dSprites, SmallNORB, Cars3D, and Shapes3D

disentanglement_lib was created by Olivier Bachem and Francesco Locatello at Google Brain Zurich for the large-scale empirical study

> [**Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations.**](https://arxiv.org/abs/1811.12359)
> *Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Rätsch, Sylvain Gelly, Bernhard Schölkopf, Olivier Bachem*. arXiv preprint, 2018.

The code is tested with Python 3 and is meant to be run on Linux systems (such as a [Google Cloud Deep Learning VM](https://cloud.google.com/deep-learning-vm/docs/)).
It uses TensorFlow, Scipy, Numpy, Scikit-Learn, TFHub and Gin.

## How does it work?
disentanglement_lib consists of several different steps:

* **Model training**: Trains a TensorFlow model and saves trained model in a TFHub module.
* **Postprocessing**: Takes a trained model, extracts a representation (e.g. by using the mean of the Gaussian encoder) and saves the representation function in a TFHub module.
* **Evaluation**: Takes a representation function and computes a disentanglement metric.
* **Visualization**: Takes a trained model and visualizes it.

All configuration details and experimental results of the different steps are saved and propagated along the steps (see below for a description).
At the end, they can be aggregated in a single JSON file and analyzed with Pandas.


## Usage
### Installing disentanglement_lib
First, clone this repository with

```
git clone https://github.com/google-research/disentanglement_lib.git
```

Then, navigate to the repository (with `cd disentanglement_lib`) and run

```
pip install .[tf_gpu]
```

(or `pip install .[tf]` for TensorFlow without GPU support).
This should install the package and all the required dependencies.
To verify that everything works, simply run the test suite with

```
dlib_tests
```

### Downloading the data sets
To download the data required for training the models, navigate to any folder and run

```
dlib_download_data
```

which will install all the required data files (except for Shapes3D which is not
publicly released) in the current working directory.
For convenience, we recommend to set the environment variable `DISENTANGLEMENT_LIB_DATA` to this path, for example by adding

```
export DISENTANGLEMENT_LIB_DATA=<path to the data directory>
```
to your `.bashrc` file. If you choose not to set the environment variable `DISENTANGLEMENT_LIB_DATA`, disentanglement_lib will always look for the data in your current folder.

### Reproducing prior experiments

To fully train and evaluate one of the 12'600 models in the paper [*Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*](https://arxiv.org/abs/1811.12359), simply run

```
dlib_reproduce --model_num=<?>
```

where `<?>` should be replaced with a model index between 0 and 12'599 which
corresponds to the ID of which model to train.
This will take a couple of hours and add a folder `output/<?>` which contains the trained model (including checkpoints and TFHub modules), the experimental results (in JSON format) and visualizations (including GIFs).
To only print the configuration of that model instead of training, add the flag `--only_print`.

After having trained several of these models, you can aggregate the results by running
the following command (in the same folder)

```
dlib_aggregate_results
```
which creates a `results.json` file with all the aggregated results.


### Running different configurations
Internally, disentanglement_lib uses [gin](https://github.com/google/gin-config) to configure hyperparameters and other settings.
To train one of the provided models but with different hyperparameters, you need to write a gin config such as `examples/model.gin`.
Then, you may use the following command

```
dlib_train --gin_config=examples/model.gin --model_dir=<model_output_directory>
```
to train the model where `--model_dir` specifies where the results should be saved.

To evaluate the newly trained model consistent with the evaluation protocol in the paper [*Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*](https://arxiv.org/abs/1811.12359), simply run

```
dlib_reproduce --model_dir=<model_output_directory> --output_directory=<output>
```
Similarly, you might also want to look at `dlib_postprocess` and `dlib_evaluate` if you want to customize how representations are extracted and evaluated.

### Starting your own research
disentanglement_lib is easily extendible and can be used to implement new models and metrics related to disentangled representations.
To get started, simply go through `examples/example.py` which shows you how to create your own disentanglement model and metric and how to benchmark them against existing models and metrics.



## Frequently asked questions

### How do I make pretty GIFs of my models?
If you run `dlib_reproduce`, they are automatically saved to the `visualizations` subfolder in your output directory. Otherwise, you can use the script `dlib_visualize_dataset` to generate them or call the function `visualize(...)` in `disentanglement_lib/visualize/visualize_model.py`.

### How are results and models saved?
After each of the main steps (training/postprocessing/evaluation), an output directory is created.
For all steps, there is a `results` folder which contains all the configuration settings and experimental results up to that step.
The `gin` subfolder contains the operative gin config for each step in the gin format.
The `json` subfolder contains files with the operative gin config and the experimental results of that step but in JSON format.
Finally, the `aggregate` subfolder contains aggregated JSON files where each file contains both the configs and results from all preceding steps.

The training step further saves the TensorFlow checkpoint (in a `tf_checkpoint` subfolder) and the trained model as a TFHub module (in a `tfhub` subfolder). Similarly, the postprocessing step saves the representation function as a TFHub module (in a `tfhub` subfolder).
If you run `dlib_reproduce`, it will create subfolders for all the different substeps that you ran. In particular, it will create an output directory for each metric that you computed.

### How do I access the results?

To access the results, first aggregate all the results using `dlib_aggregate_results` by specifying a glob pattern that captures all the results files.
For example, after training a couple of different models with `dlib_reproduce`, you would specify

```
dlib_aggregate --output_path=<...>.json \
  --result_file_pattern=<...>/*/metrics/*/*/results/aggregate/evaluation.json
```
The first * in the glob pattern would capture the different models, the second * different representations and the last * the different metrics.
Finally, you may access the aggregated results with:

```python
from disentanglement_lib.utils import aggregate_results
df = aggregate_results.load_aggregated_json_results(output_path)
```
## Where to look in the code?

The following provides a guide to the overall code structure:

**(1) Training step:**

* `disentanglement_lib/methods/unsupervised`:
Contains the training protocol (`train.py`) and all the model functions
for training the methods (`vae.py`). The methods all inherit from the
`GaussianEncoderModel` class.
* `disentanglement_lib/methods/shared`:
 Contains shared architectures, losses, and optimizers used in the different models.

**(2) Postprocessing step:**

* `disentanglement_lib/postprocess`:
Contains the postprocessing pipeline (`postprocess.py`) and the two extraction methods (`methods.py`).

**(3) Evaluation step:**

* `disentanglement_lib/evaluation`: Contains the evaluation protocol (`evaluate.py`).

* `disentanglement_lib/evaluation/metrics`:
Contains implementation of the different disentanglement metrics.

**Hyperparameters and configuration files:**

* `disentanglement_lib/config/unsupervised_study_v1`:
Contains the gin configuration files (`*.gin`) for the different steps as well as the hyperparameter sweep (`sweep.py`) for the experiments in the paper *Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*.

**Shared functionality:**

* `bin`: Scripts to run the different pipelines, visualize the data sets as well as the models and aggregate the results.

* `disentanglement_lib/data/ground_truth`:
Contains all the scripts used to generate the data. All the datasets (in `named_data.py`) are instances of the class `GroundTruthData}`.

* `disentanglement_lib/utils`:
Contains helper functions to aggregate and save the results of the pipeline as well as the trained models.

* `disentanglement_lib/visualize`:
Contains visualization functions for the datasets and the trained models.


## Feedback
Please send any feedback to bachem@google.com and francesco.locatello@tuebingen.mpg.de.

## Citation
If you use **disentanglement_lib**, please consider citing:

```
@article{locatello2018challenging,
  title={Challenging common assumptions in the unsupervised learning of disentangled representations},
  author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
  journal={arXiv preprint arXiv:1811.12359},
  year={2018}
}
```

### This is not an officially supported Google product.


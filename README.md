# Differential Privacy Generative Models Framework

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-differential-privacy-generative-models-framework)](https://api.reuse.software/info/github.com/SAP-samples/security-research-differential-privacy-generative-models-framework)
[![Not Maintained](https://img.shields.io/badge/Maintenance%20Level-Not%20Maintained-yellow.svg)](https://gist.github.com/cheerfulstoic/d107229326a01ff0f333a1d3476e068d)


## Description
The dp-generative-models repository is a framework for training and evaluating differentially private generatative models.
It allows to create an anonymized version of an original dataset potentially containing personal data. If that is the case, please make sure to have a valid legal ground for processing that data in the first place. Next, make sure that anonymization is compatible with the purposes the data has been collected for. Finally, remove all direct identifiers from the original dataset before using this tool.

### Code Formatting

This repository uses Black for code formatting. When you commit, Black will re-format your files and you will be asked to once again add the files before you are able to commit and push. Ensuring a standard code style is useful because it eliminates formatting changes in file differences. 

### Best Practice

The goal of the framework is to enable a user to systematically identify the best model configuration for a given data generation task. This is done by running an experiment, evaluating the results, and then running a new experiment with ideally better parameters. Fundamentally, determining the best parameter configuration is an optimization procedure over the output of data utility/privacy tests. This means that if more than one experiment is run the test scores are really validation scores, and, subsequently, they will exhibit a positive bias after a number iterative experiments. It is therefore recommended that prior to running any experiments the user remove a portion of the data, about 10%, to use as a holdout set after the best parameter configuration is identified. That way, an unbiased appoximation of the generated data's utility/privacy can be determined. 

### Running an Experiment

The procedure of running an experiment can be summarized as:

  1. Check if an experiment configuration run already exists and whether to overwrite.
  2. Pre-process the data, determine the train/test indices, and transform the data to be suitable for model training.
  3. Train the model.
  4. Sample data from the model.
  5. Verify the integrity of the output data and save the result.

The framework takes care of steps 1, 2, and 5, as well as controls the overall workflow and aims to provide functionality that simplifies steps 3 and 4. In order to run an experiment, the necessary actions are to create a subclass of Experiment in the form of YourModelExperiment. These should be stored in the model_experiments directory. The two methods that must be implemented within the new class are: 

`fit_model`: This is where the entire training procedure for your model should take place (e.g. looping over the training data, calculating the loss, applying the gradients, recording epsilon, etc). You will have access to the experiment config that contains the model, model training, and differential privacy configurations. Additionally, the Experiment class contains methods, such as  `record_epsilon` and `do_early_stop`, that can be useful during training.

`sample_model`: This is where the fake data should be generated. The attribute `pre_fit_transformer` may come in handy to conduct an inverse transformation that converts the generated data back to the same format as the original data. 

Of course, you must implement the actual model to be referenced in YourModelExperiment if it not already included in the models directory. Any new model should be added to the GenerativeModelType enum class and GENERATIVE_MODEL_MAP. 

The following dictionary below is an example experiment configuration for a VAE. Any parameter that is specified by the user will overwrite the default if it exists or be added to the config if it does not. In YourModelExperiment, the config can be accessed with the attribute `self.config`. It is required to pass in a dataset, model type, and model config. A detailed documentation for each of these parameters can be found in the docstring of the Experiment class in general/experiment.py. 

In order to run an experiment with the same configuration multiple times, the `name` parameter needs to be updated for each run (e.g. changing `vae_experiment` to `vae_experiment_v2`. Alternatively, you can change the `seed` value for each run, which will enable the same experiment configuration to have multiple trials with reproducible results.

```
config = {
    "dataset": "uci_credit_card.csv",
    "model_type": GenerativeModelType.VAE,
    "name": "vae_experiment",
    "data_path": "../data/",
    "data_processing": {
        "categorical_columns": [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_1",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "DEFAULT",
        ],
    },
    "model_config": {
        "compress_dims": [128, 128],
        "decompress_dims": [128, 128],
        "compress_activations": [Activation.LeakyReLU, Activation.LeakyReLU],
        "decompress_activations": [Activation.LeakyReLU, Activation.LeakyReLU],
        "latent_dim": 128,
        "batch_norm": True,
        "output_activations": {
            "continuous": OutputActivation.TANH,
            "categorical": OutputActivation.SOFTMAX,
        },
    },
    "model_train": {
        "seed": 1,
        "test_pct": 0.3,
        "k_fold": False,
        "stratified_by_col": None,
        "epochs": 50,
        "batch_size": 64,
        "dp_optimizer_type": OptimizerType.DPAdamGaussianOptimizer,
        "record_gradients": {"enabled": True, "subset": ["DEFAULT", "SEX"],},
        "transformer": {
            "type": TransformerType.BGM,
            "kwargs": {"outlier_clipping": True, "n_clusters": 6, "cont_feature_range": (-1, 1)},
        },
        "early_stop": {"enabled": False},
        "early_stop_epsilon": {"enabled": True, "value": 2.5},
    },
    "diff_priv": {
        "enabled": True,
        "microbatches": 1,
        "l2_norm_clip": 5.0,
        "noise_multiplier": 0.8,
    },
}
```

Using the above config, an experiment can be run with the following two lines:

```
vae_experiment = VaeExperiment(config)
vae_experiment.run()
```

It is important to note that although transformers and models are independent, it generally makes sense to scale continuous features within the same range as the output activation that will be applied to them. For example, the config above uses the parameter `cont_feature_range` to specify that the original data should be scaled between -1 and 1, which means that tanh is a suitable continuous feature output activation function to apply.

### Evaluating an Experiment

Upon completion of an experiment, the file name of the output data will be printed. The hash is unique to the config.

```
step 0: mean loss = 30.27
step 50: mean loss = 29.21
epsilon = 1.56
WARNING:root:continuous cols with 0 values. Percent equal to 0: {'capital-gain': 0.0012285012285012285, 'capital-loss': 0.03470515970515971}
experiment saved to e8fcc84f68d181834a923ef575ce0d0f0a936fe0.pkl
```

The quality of the fake data can then be accessed by evaluating it within MachineLearningEvaluator.

```
eval_input = {
    'target_col': 'DEFAULT',
    'data_path': '../data/',
    'config_file': 'e8fcc84f68d181834a923ef575ce0d0f0a936fe0.pkl'
}

evaluator = MachineLearningEvaluator(**eval_input)
```

Alternatively, if you are seeking to access a config but do not remember the file name, you can use the method `existing_configs` in order to see all the experiment outputs that exist in your data path containing an experiment name or certain config parameters. 

```
name = 'my_vae_experiment'

test_config = {
    'dataset': 'uci_credit_card.csv',
    'name': 'vae_experiment',
    'diff_priv': {"enabled": True},
    'model_config': {
        'compress_dims': [128, 128],
        'decompress_dims': [128, 128]
     }
}

existing_configs = MachineLearningEvaluator.existing_configs(data_path='../data/', test_config=test_config)
 ```
 
 #### Extracting the Generated Data
 
After instantiating the evaluator with a config, the fake data can be output using the `fake_df` method of the evaluator class. The model_run parameter only needs to be indicated if the experiment used k_fold and you would like the fake data from not the first fold. 
 
 ```
 fake_df = evaluator.fake_df(model_run=0) 
 ```


 #### Evaluate Distributions and Correlations of Generated Data
 
 ```
 evaluator.plot_distributions()
 evaluator.plot_heatmap()
 ```
 
 #### Evaluating the Generated Data's Utility
 
The quality of generated data can be evaluated with the following strategy:
  
  1. Train a model using real train data.
  2. Train a model using fake data.
  3. Predict a test set of real data using both models.
  4. Compare the quality of the predictions.
 
The following method will output the results of this test for each of the model runs. Remember, the model runs will be greater than 1 if k_fold=True. Any sklearn estimator, metric(s), and scaler can be passed. If they are not already handled, they can be be added to the EvaluatorModelType, MetricType, and ScalerType enum classes. 
 
 ```
model_kwargs = {'n_estimators': 50, 'max_depth': 5}
   
scores = evaluator.data_utility_scores(
    EvaluatorModelType.RandomForestClassifier, 
    [MetricType.F1, MetricType.Accuracy],
    ScalerType.ROBUST,
    model_kwargs=model_kwargs)
print(scores)
```

After completing all experiments and identifying the best parameter configuration, the data's utility can be evaluated on the holdout set. The first step is to run an experiment to generate data from the entire non-holdout dataset. The parameters that should be changed are `test_pct` to `0` and `k_fold` to `False` in the `model_train` config. Once the training is complete and the experiment is saved, the holdout score can be evaluated by passing the holdout set to `data_utility_scores`.

```
eval_input = {
    'target_col': 'DEFAULT',
    'data_path': '../data/',
    'config_file': '672ae392cf493f645ffd67ae64bccec71a3b2577.pkl' (new experiment hash)
}

evaluator = MachineLearningEvaluator(**eval_input)

holdout = pd.read_csv("uci_credit_card_holdout.csv").values

scores = evaluator.data_utility_scores(
    EvaluatorModelType.RandomForestClassifier, 
    [MetricType.F1, MetricType.Accuracy],
    ScalerType.ROBUST,
    model_kwargs=model_kwargs
    holdout=holdout
)
print(scores)
```

### References

Tensorflow privacy: https://github.com/tensorflow/privacy

SDGym: https://github.com/sdv-dev/SDGym

Black: https://github.com/psf/black

## Requirements
See requirements file.

## Download and Installation
Clone the repository:

`git clone https://github.com/SAP-samples/security-research-differential-privacy-generative-models-framework.git`

Install the requirements to your local environment:

`pip install -r requirements.txt`

## Known Issues
None.

## How to obtain support

[Create an issue](https://github.com/SAP-samples/security-research-differential-privacy-generative-models-framework/issues) in this repository if you find a bug or have questions about the content.

For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## License
Copyright (c) 2021 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.

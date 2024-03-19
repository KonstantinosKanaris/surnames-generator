* [Overview](#Overview)
* [Key Features](#Key--Features)
* [Development](#Development)
* 

## Overview 🔍
A Python project with PyTorch for generating surnames from various nationalities. 
Implements 2 high-level processes:
- Training of character-based text generation PyTorch models
- New surnames generation with a pre-trained model

Both processes are fully configurable through dedicated configuration files. For 
instance, for the training process, users can define training hyperparameters, such 
as number of epochs, batch size, model/optimizer settings, in a YAML file. This allows for 
easy customization and experimentation with different configurations without modifying 
the source code.

Currently, the project supports three character-based text generation models, i.e.,
``simple_rnn_generator``, ``gru_generator``, ``lstm_generator``, but by leveraging the
benefits of the Factory Design Pattern for selection and initialization of different
models, the project can be easily extended to support more architectures. 

Additionally, all the implemented models have 2 modes, i.e., unconditioned generation 
(predict characters to form random surnames), and conditioned generation (predict characters
to form surnames biased towards a specific nationality).

The project also supports running multiple training experiments using different models 
and/or  other hyperparameters. While the framework can handle hyperparameter tuning, its 
primary objective is to define optimal hyperparameters for different models, facilitating 
comparison of these models  on the same data.

### Project Structure 🌲
```
surnames-generator
├── .gitignore
├── .pre-commit-config.yaml     | Pre-commit hooks
├── Makefile                    | Development commands for code formatting, linting, etc
├── Pipfile                     | Project's dependencies and their versions using pipenv format
├── Pipfile.lock                | Auto-generated file that locks the dependencies to specific versions for reproducibility
├── README.md
├── colours.mk                  | A Makefile fragment containing color codes for terminal output styling
├── configs
│   ├── evaluation.yaml   | Configuration parameters for the generation of new surnames process
│   └── experiments.yaml  | Configuration parameters for the training process. May contains multiple training experiments
├── docs                        | Sphinx generated documentation
│   ├──build
│   ├──source
│   └── ...
├── mypy.ini                    | Configuration file for the MyPy static type checker
└── surnames_generator          | The main Python package containing the project's source code
    ├── __about__.py            | Metadata about the project, i.e., version number, author information, etc
    ├── __init__.py     
    ├── __main__.py             | The entry point for running the package as a script. Calls one of the controllers
    ├── controllers.py          | Contains the training and the generation controller.
    ├── datasets.py              
    ├── engine
    │   ├── __init__.py
    │   ├── generator.py  | Contains the new surnames generation process
    │   ├── trainer.py    | Contains the training process
    │   └── utils.py      | Auxilliary functions/classes for the training process such as EarlyStopping
    ├── factories
    │   ├── __init__.py
    │   ├── client.py    | Interacts with the factories to return different instances of models, optimizers, and loss functions
    │   └── factories.py | Contains factory classes for creating different models, optimizers, and loss functions
    ├── logger
    │   └── logging.ini  | Configuration file for Python's logging module
    ├── models.py              | Contains the `simple_rnn_generator`, `gru_generator`, and `lstm_generator` models
    └── utils
        ├── __init__.py
        ├── aux.py             | Auxilliary functions/classes used across the project
        ├── vectorizers.py     | Vectorizes text sequences to numbers. Character-based vectorization.
        └── vocabulary.py      | Maps tokens to indices and vice versa
```

## Key Features 🔑
* **Customizable Experiments**: Define multiple experiments easily by configuring model
architecture, optimizer, learning rate, batch size, and other hyperparameters in a YAML
configuration file
* **Customizable Models**: Easily integrate new character-based text generation PyTorch 
models, allowing for seamless experimentation with novel architectures and configurations
* **Experiment Tracking**: Utilize MLFlow for tracking training process
* **Checkpointing**: Ensure training progress is saved with checkpointing functionality,
allowing for easy resumption of training from the last saved state
* **EarlyStopping**: Automatically stop training when the model's performance stops
improving on a validation set



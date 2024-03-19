import argparse

from surnames_generator import __title__, logger
from surnames_generator.controllers import (
    SurnamesGeneratorController,
    TrainingController,
)
from surnames_generator.datasets import SurnamesDataset
from surnames_generator.utils.aux import load_yaml_file


def parse_arguments() -> argparse.Namespace:
    """
    Constructs parsers and subparsers.

    Returns:
        argparse.Namespace:
            The parser/subparser with its arguments.
    """
    parser = argparse.ArgumentParser(
        description=f"Command Line Interface for {__title__}"
    )

    subparsers = parser.add_subparsers(
        description="Project functionalities", dest="mode"
    )

    train = subparsers.add_parser(
        name="train", help="This is the subparser for training."
    )
    generate = subparsers.add_parser(
        name="generate", help="This is the subparser for generating new surnames."
    )

    train.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for training."
        "Contains data paths, hyperparameters, etc.",
    )
    train.add_argument(
        "--resume_from_checkpoint",
        type=str,
        choices={"yes", "no"},
        required=False,
        default="no",
        help="If `yes` the training will resume from the last saved checkpoint.",
    )

    generate.add_argument(
        "--config",
        type=str,
        required=True,
        help="The config file (yaml) required for generating new surnames."
        "Contains the data and model paths, and model's initialization parameters.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.mode == "train":
        config = load_yaml_file(filepath=arguments.config)
        dataset = SurnamesDataset.load_dataset_from_csv(
            surnames_csv=config["data_path"]
        )

        for i, experiment in enumerate(config["experiments"]):
            logger.info(f"Experiment {i+1}")
            training_controller = TrainingController(
                dataset=dataset,
                hyperparameters=experiment,
                checkpoints_dir=config["checkpoints_dir"],
            )
            training_controller.prepare_and_start_training()
    elif arguments.mode == "generate":
        config = load_yaml_file(filepath=arguments.config)
        dataset = SurnamesDataset.load_dataset_from_csv(
            surnames_csv=config["data_path"]
        )
        vectorizer = dataset.get_vectorizer()
        generator_controller = SurnamesGeneratorController(
            model_path=config["model_path"],
            model_name=config["model_name"],
            model_init_params=config["model_init_params"],
            vectorizer=vectorizer,
            num_samples=config["num_samples"],
        )
        generator_controller.generate_surnames()
    else:
        logger.error("Not supported mode.")

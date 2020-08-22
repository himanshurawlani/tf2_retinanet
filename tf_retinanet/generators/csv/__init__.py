"""
Copyright 2017-2019 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
	http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import get_csv_generator
from .eval import get_csv_evalutaion_callback
from tf_retinanet.utils.config import set_defaults


default_config = {
    "train_annotations_path": None,
    "train_classes_path": None,
    "val_annotations_path": None,
    "val_classes_path": None,
    "test_annotations_path": None,
    "test_classes_path": None,
    "mask": False,
}


def from_config(config, submodels_manager, preprocess_image, **kwargs):
    """ Return generators and submodels as indicated in the config.
		The number of classes (necessary for creating the classification submodel)
		is taken from the CSV generators. Hence, submodels can be initialized only after the generators.
	Args
		config : Dictionary containing information about the generators.
				 It should contain:
					annotations_path    : Path to the directory where the annotations CSV is stored.
					classes_path        : Path to the directory where the classes CSV is stored.
				 If not specified, default values indicated above will be used.
		submodel_manager : Class that handles and initializes the submodels.
		preprocess_image : Function that describes how to preprocess images in the generators.
	Return
		generators : Dictionary containing generators and evaluation procedures.
		submodels  : List of initialized submodels.
	"""
    # Set default configuration parameters.
    config = set_defaults(config, default_config)

    # If no annotations dir is set, ask the user for it.
    if ("train_annotations_path" not in config) or not config["train_annotations_path"]:
        config["train_annotations_path"] = input(
            "Please input the train annotations CSV folder:"
        )
    if ("train_classes_path" not in config) or not config["train_classes_path"]:
        config["train_classes_path"] = input(
            "Please input the train classes CSV folder:"
        )

    generators = {}

    # We should get the number of classes from the generators.
    num_classes = 0

    # Get the generator that supports masks if needed.
    if config["mask"]:
        from tf_maskrcnn_retinanet.generators import Generator
    else:
        from tf_retinanet.generators import Generator
    CSVGenerator = get_csv_generator(Generator)

    # If needed, get the annotations generator.
    if (
        config["train_annotations_path"] is not None
        and config["train_classes_path"] is not None
    ):
        generators["train"] = CSVGenerator(
            config, preprocess_image, config["train_annotations_path"], config["train_classes_path"]
        )

    # If needed, get the validation generator.
    if config["val_annotations_path"] is not None:
        generators["validation"] = CSVGenerator(
            config, preprocess_image, config["val_annotations_path"], config["train_classes_path"]
        )

    # If needed, get the validation generator.
    if config["test_annotations_path"] is not None:
        generators["test"] = CSVGenerator(
            config, preprocess_image, config["test_annotations_path"], config["train_classes_path"]
        )

    # Disable the transformations after getting the CSV generator.
    config["transform_generator_class"] = None
    config["visual_effect_generator_class"] = None

    generators['evaluation_callback']  = get_csv_evalutaion_callback()

    # Set up the submodels for this generator.
    assert generators["train"].num_classes != 0, "Got 0 classes from CSV generator."

    # Instantiate the submodels for this generator.
    submodels_manager.create(num_classes=generators["train"].num_classes())

    return generators, submodels_manager.get_submodels()

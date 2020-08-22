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

from .generator import Generator  # noqa: F401

from ..utils import import_package


def preprocess_config(config):
	""" Converts specified config entries to the desired classes or functions.
	Args
		config: Dictionary containing generator details.
	Returns
		The config dictionary extended with the desired classes and functions.
	"""
	# Set the tranform generator class. If the transform_generator flag is set to basic, use only flip_x.
	if config['transform_generator']  == 'basic':
		from ..utils.transform import random_transform_generator
		config['transform_generator_class'] = random_transform_generator(flip_x_chance=0.5)
	elif config['transform_generator']  == 'random':
		from ..utils.transform import random_transform_generator
		config['transform_generator_class'] = random_transform_generator(
			min_rotation=-0.1,
			max_rotation=0.1,
			min_translation=(-0.1, -0.1),
			max_translation=(0.1, 0.1),
			min_shear=-0.1,
			max_shear=0.1,
			min_scaling=(0.9, 0.9),
			max_scaling=(1.1, 1.1),
			flip_x_chance=0.5,
			flip_y_chance=0.5,
		)
	else:
		config['transform_generator_class'] = None

	# If the visual_effect_generator flag is set to random, set it to the random preset.
	if config['visual_effect_generator']  == 'random':
		from ..utils.image import random_visual_effect_generator
		config['visual_effect_generator_class'] = random_visual_effect_generator(
			contrast_range=(0.9, 1.1),
			brightness_range=(-.1, .1),
			hue_range=(-0.05, 0.05),
			saturation_range=(0.95, 1.05)
		)
	else:
		config['visual_effect_generator_class'] = None

	# If the transform_parameters flag is set to default, use TranformParameters.
	if config['transform_parameters']  == 'standard':
		from ..utils.image import TransformParameters
		config['transform_parameters_class'] = TransformParameters()
	else:
		config['transform_parameters_class'] = None

	return config


def get_generators(config, submodels_manager, preprocess_image, **kwargs):
	""" Imports generators from an external package,
		and with the retrieved information the submodels manager creates the sumbodels.
		The link between submodels and generators depends on the used generator, hence they are created together in the external package.
	Args
		config:            Dictionary containing name and details for importing the generator.
		submodels_manager: Manager containing details for the creation of the submodels.
		preprocess_image:  Function used to preprocess images in the generator.
	Returns
		The specified generators and submodels.
	"""
	generator_pkg = import_package(config['name'], 'tf_retinanet.generators')

	return generator_pkg.from_config(
		preprocess_config(config['details']),
		submodels_manager,
		preprocess_image,
		**kwargs
	)

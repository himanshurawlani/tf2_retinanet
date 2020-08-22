import os

import tensorflow as tf

from tf_retinanet.backbones        import Backbone
from tf_retinanet.models.retinanet import retinanet
from tf_retinanet.utils.image      import preprocess_image
from .resnet50v2 import ResnetV2, PaddingFromKernelSize, BottleneckV2Unit, StandardizedConv2D


class ResNet50V2Backbone(Backbone):
	""" Describes backbone information and provides utility functions.
	"""

	def __init__(self, config):
		super(ResNet50V2Backbone, self).__init__(config)
		# TODO: Update custom objects dictionary here
		self.custom_objects.update({'PaddingFromKernelSize': PaddingFromKernelSize})
		self.custom_objects.update({'BottleneckV2Unit': BottleneckV2Unit})
		self.custom_objects.update({'StandardizedConv2D': StandardizedConv2D})

	def retinanet(self, *args, **kwargs):
		""" Returns a retinanet model using the correct backbone.
		"""
		return resnet50v2_retinanet(*args, weights=self.weights, modifier=self.modifier, **kwargs)

	def validate(self):
		""" Checks whether the backbone string is correct.
		"""
		allowed_backbones = ['resnet50v2']
		backbone = self.backbone.split('_')[0]

		if backbone not in allowed_backbones:
			raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

	def preprocess_image(self, inputs):
		""" Takes as input an image and prepares it for being passed through the network.
		"""
		# Default preprocessing for ResnetV2 in keras_applications.
		return preprocess_image(inputs, mode='tf')

def resnet50v2_retinanet(submodels, inputs=None, modifier=None, weights='imagenet', **kwargs):
	""" Creates a retinanet model using the ResNet50v2 backbone.
	Arguments
		submodels: RetinaNetSubmodels.
		inputs:    The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
		modifier:  A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).
		weights:   Weights for the backbone (default is imagenet weights).
	Returns:
		RetinaNet model with ResNet50v2 backbone.
	"""
	KNOWN_MODELS = {
			f'{bit}-R{l}x{w}': f'gs://bit_models/{bit}-R{l}x{w}.h5'
			for bit in ['BiT-S', 'BiT-M']
			for l, w in [(50, 1), (50, 3), (101, 1), (101, 3), (152, 4)]
	}

	NUM_UNITS = {
			k: (3, 4, 6, 3) if 'R50' in k else
				(3, 4, 23, 3) if 'R101' in k else
				(3, 8, 36, 3)
			for k in KNOWN_MODELS
	}
	
	args_model = "BiT-M-R50x1"
	args_bit_pretrained_dir = 'pretrained_models'

	# Choose default input.
	if tf.keras.backend.image_data_format() == 'channels_first':
		inputs = tf.keras.Input(shape=(3, None, None))
		input_shape = (None, 3, None, None)
	else:
		inputs = tf.keras.Input(shape=(None, None, 3))
		input_shape = (None, None, None, 3)

	# Create the resnet backbone.
	resnet = ResnetV2(
			num_units=NUM_UNITS[args_model],
			model_input=inputs,
			num_outputs=None,
			filters_factor=int(args_model[-1])*4,
			name="resnet",
			trainable=True,
			dtype=tf.float32)

	tf.io.gfile.makedirs(args_bit_pretrained_dir)
	bit_model_file = os.path.join(args_bit_pretrained_dir, f'{args_model}.h5')
	if not tf.io.gfile.exists(bit_model_file):
		model_url = KNOWN_MODELS[args_model]
		print(f'Downloading the model from {model_url}...')
		tf.io.gfile.copy(model_url, bit_model_file)

	# Create weights for the layers 
	resnet.build(input_shape)
	resnet.load_weights(bit_model_file, by_name=True)
	# Since we are renaming layers we call get_connected_model() after loading weights
	resnet = resnet.get_connected_model()

	# Invoke modifier if given.
	if modifier:
		resnet = modifier(resnet)

	# Get output layers.
	layer_names = ["block2_unit4", "block3_unit6", "block4_unit3"]
	layer_outputs = [resnet.get_layer(name).output for name in layer_names]

	return retinanet(resnet.inputs, layer_outputs, submodels, **kwargs)

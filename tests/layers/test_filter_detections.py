import tensorflow as tf
import tf_retinanet.layers

import numpy as np


class TestFilterDetections(object):
	def test_simple(self):
		# Create simple FilterDetections layer.
		filter_detections_layer = tf_retinanet.layers.FilterDetections()

		# Create simple input.
		boxes = np.array([[
			[0, 0, 10, 10],
			[0, 0, 10, 10],  # This will be suppressed.
		]], dtype=tf.keras.backend.floatx())
		boxes = tf.keras.backend.constant(boxes)

		classification = np.array([[
			[0, 0.9],  # This will be suppressed.
			[0, 1],
		]], dtype=tf.keras.backend.floatx())
		classification = tf.keras.backend.constant(classification)

		# Compute output
		actual_boxes, actual_scores, actual_labels = filter_detections_layer.call([boxes, classification])
		actual_boxes  = tf.keras.backend.eval(actual_boxes)
		actual_scores = tf.keras.backend.eval(actual_scores)
		actual_labels = tf.keras.backend.eval(actual_labels)

		# define expected output
		expected_boxes = -1 * np.ones((1, 300, 4), dtype=tf.keras.backend.floatx())
		expected_boxes[0, 0, :] = [0, 0, 10, 10]

		expected_scores = -1 * np.ones((1, 300), dtype=tf.keras.backend.floatx())
		expected_scores[0, 0] = 1

		expected_labels = -1 * np.ones((1, 300), dtype=tf.keras.backend.floatx())
		expected_labels[0, 0] = 1

		# assert actual and expected are equal
		np.testing.assert_array_equal(actual_boxes, expected_boxes)
		np.testing.assert_array_equal(actual_scores, expected_scores)
		np.testing.assert_array_equal(actual_labels, expected_labels)

	def test_simple_with_other(self):
		# create simple FilterDetections layer
		filter_detections_layer = tf_retinanet.layers.FilterDetections()

		# create simple input
		boxes = np.array([[
			[0, 0, 10, 10],
			[0, 0, 10, 10],  # this will be suppressed
		]], dtype=tf.keras.backend.floatx())
		boxes = tf.keras.backend.constant(boxes)

		classification = np.array([[
			[0, 0.9],  # this will be suppressed
			[0, 1],
		]], dtype=tf.keras.backend.floatx())
		classification = tf.keras.backend.constant(classification)

		other = []
		other.append(np.array([[
			[0, 1234],  # this will be suppressed
			[0, 5678],
		]], dtype=tf.keras.backend.floatx()))
		other.append(np.array([[
			5678,  # this will be suppressed
			1234,
		]], dtype=tf.keras.backend.floatx()))
		other = [tf.keras.backend.constant(o) for o in other]

		# compute output
		actual = filter_detections_layer.call([boxes, classification] + other)
		actual_boxes  = tf.keras.backend.eval(actual[0])
		actual_scores = tf.keras.backend.eval(actual[1])
		actual_labels = tf.keras.backend.eval(actual[2])
		actual_other  = [tf.keras.backend.eval(a) for a in actual[3:]]

		# define expected output
		expected_boxes = -1 * np.ones((1, 300, 4), dtype=tf.keras.backend.floatx())
		expected_boxes[0, 0, :] = [0, 0, 10, 10]

		expected_scores = -1 * np.ones((1, 300), dtype=tf.keras.backend.floatx())
		expected_scores[0, 0] = 1

		expected_labels = -1 * np.ones((1, 300), dtype=tf.keras.backend.floatx())
		expected_labels[0, 0] = 1

		expected_other = []
		expected_other.append(-1 * np.ones((1, 300, 2), dtype=tf.keras.backend.floatx()))
		expected_other[-1][0, 0, :] = [0, 5678]
		expected_other.append(-1 * np.ones((1, 300), dtype=tf.keras.backend.floatx()))
		expected_other[-1][0, 0] = 1234

		# assert actual and expected are equal
		np.testing.assert_array_equal(actual_boxes, expected_boxes)
		np.testing.assert_array_equal(actual_scores, expected_scores)
		np.testing.assert_array_equal(actual_labels, expected_labels)

		for a, e in zip(actual_other, expected_other):
			np.testing.assert_array_equal(a, e)

	def test_mini_batch(self):
		# create simple FilterDetections layer
		filter_detections_layer = tf_retinanet.layers.FilterDetections()

		# create input with batch_size=2
		boxes = np.array([
			[
				[0, 0, 10, 10],  # this will be suppressed
				[0, 0, 10, 10],
			],
			[
				[100, 100, 150, 150],
				[100, 100, 150, 150],  # this will be suppressed
			],
		], dtype=tf.keras.backend.floatx())
		boxes = tf.keras.backend.constant(boxes)

		classification = np.array([
			[
				[0, 0.9],  # this will be suppressed
				[0, 1],
			],
			[
				[1,   0],
				[0.9, 0],  # this will be suppressed
			],
		], dtype=tf.keras.backend.floatx())
		classification = tf.keras.backend.constant(classification)

		# compute output
		actual_boxes, actual_scores, actual_labels = filter_detections_layer.call([boxes, classification])
		actual_boxes  = tf.keras.backend.eval(actual_boxes)
		actual_scores = tf.keras.backend.eval(actual_scores)
		actual_labels = tf.keras.backend.eval(actual_labels)

		# define expected output
		expected_boxes = -1 * np.ones((2, 300, 4), dtype=tf.keras.backend.floatx())
		expected_boxes[0, 0, :] = [0, 0, 10, 10]
		expected_boxes[1, 0, :] = [100, 100, 150, 150]

		expected_scores = -1 * np.ones((2, 300), dtype=tf.keras.backend.floatx())
		expected_scores[0, 0] = 1
		expected_scores[1, 0] = 1

		expected_labels = -1 * np.ones((2, 300), dtype=tf.keras.backend.floatx())
		expected_labels[0, 0] = 1
		expected_labels[1, 0] = 0

		# assert actual and expected are equal
		np.testing.assert_array_equal(actual_boxes, expected_boxes)
		np.testing.assert_array_equal(actual_scores, expected_scores)
		np.testing.assert_array_equal(actual_labels, expected_labels)

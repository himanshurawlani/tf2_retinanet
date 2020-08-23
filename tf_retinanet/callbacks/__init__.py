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

from .common import RedirectModel

import os
import numpy as np

import tensorflow as tf


def get_callbacks(
    config,
    model,
    training_model,
    prediction_model,
    validation_generator=None,
    evaluation_callback=None,
    earlystopping=None,
    tensorboard=None,
    reduceLR=None,
):
    """ Returns the callbacks indicated in the config.
	Args
		config              : Dictionary with indications about the callbacks.
		model               : The used model.
		prediction_model    : The used prediction model.
		training_model      : The used training model.
		validation_generator: Generator used during validation.
		evaluation_callback : Callback used to perform evaluation.
		earlystopping		: EarlyStopping criterion callback.
		tensorboard			: Monitor training with TensorBoard.
		reduceLR			: Reduce Learning Rate on plateau callback.
    
	Returns
		The indicated callbacks.
	"""
    callbacks = []

    tensorboard_callback=None
    # Create TensorBoard Callback.
    try:
        if config['tensorboard']:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                config["tensorboard_path"]
            )
            callbacks.append(tensorboard_callback)
    except KeyError as e:
        pass

    # Evaluate the model.
    if validation_generator:
        if not evaluation_callback:
            raise NotImplementedError("Standard evaluation_callback not implement yet.")
        evaluation_callback = evaluation_callback(validation_generator, tensorboard=tensorboard_callback)
        evaluation_callback = RedirectModel(evaluation_callback, prediction_model)
        callbacks.append(evaluation_callback)

    # Save snapshots of the model.
    try:
        os.makedirs(os.path.join(config["snapshots_path"], config["project_name"]))
    except FileExistsError as e:
        print(e)
        print("Folder already created, moving on.")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(
            config["snapshots_path"], config["project_name"], "train_model.h5"
        ),
        verbose=1,
        save_best_only=True,
        monitor="mAP",
        mode="max"
    )
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    # Create Reduce Learning Rate on Plateau Callback.
    try:
        if config['reduceLR']:
            reducer_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='mAP',
                factor=np.sqrt(0.1),
                min_lr=0.5e-6,
                patience=config["reduceLR_patience"],
                verbose=1
            )
            callbacks.append(reducer_callback)
    except KeyError as e:
        pass

    # Create earlystopping callback.
    try:
        if config['earlystopping']:
            earlystopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor="mAP", patience=config["earlystopping_patience"],  min_delta=1e-4,
                mode='max', verbose=1, restore_best_weights=True
            )
            callbacks.append(earlystopping_callback)
    except KeyError as e:
        pass
    
    return callbacks

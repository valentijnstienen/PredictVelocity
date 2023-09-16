import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

import os 
from tensorflow import keras
from math import sqrt
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from SETTINGS import *

class EarlyStopping(keras.callbacks.Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every epoch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.

    Args:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of epochs with no improvement
          after which training will be stopped.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
          displays messages when the callback takes an action.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the epoch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used. An epoch will be restored regardless
          of the performance relative to the `baseline`. If no epoch
          improves on `baseline`, training will run for `patience`
          epochs and restore weights from the best epoch in that set.

    Example:

    >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     epochs=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 epochs are run.
    4
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ["auto", "min", "max"]:
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first epoch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            self.wait = 0
        # THE CHANGE IS HERE
        if self.baseline is None or self._is_better(
            current, self.baseline
        ):
            self.wait = 0

        # Only check after the first epoch.
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        pass
    
    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
    def _is_better(self, monitor_value, reference_value):
        return monitor_value>=reference_value
        #return self.monitor_op(monitor_value - self.min_delta, reference_value)

def mixed_neural_network_SPEED(x_train_info, x_train_image, y_train, group_indices, hyper_settings, random_state, save_model = True, save_name = "best_model"):
    """
     Here we train a deep learning model to predict y_train, based on 
     images and on tabular data. 
      
    """
    if not os.path.exists("Results/"+CASENAME+"/temp_best_models"): os.makedirs("Results/"+CASENAME+"/temp_best_models")
    
    ##################################################################
    ##################### CREATE VALIDATION SET ######################
    ##################################################################
    # VALIDATION_SPLIT = 0.2

    # Half of the validation set consists of known roads and half of it of unknown roads
    number_of_samples=len(x_train_info)
    indices_roads = list(group_indices)
    random.Random(random_state).shuffle(indices_roads)
    indices_UR_VS = [item for sublist in indices_roads for item in sublist][0:int(VALIDATION_SPLIT*number_of_samples)] # Unknown roads
    #indices_KR_VS = list(y_train[~y_train.index.isin(indices_UR_VS)].sample(frac=0.5*VALIDATION_SPLIT, replace=False, random_state=random_state).index) # Known roads
    
    # Create validation and training data
    validation_indices = indices_UR_VS##indices_KR_VS + indices_UR_VS#
    #num_known = len(indices_KR_VS)
    train_indices = [e for i, e in enumerate(y_train.index) if i not in validation_indices]
    x_val_info, x_val_image, y_val = x_train_info[validation_indices], x_train_image[validation_indices], y_train[validation_indices]
    x_train_info, x_train_image, y_train = x_train_info[train_indices], x_train_image[train_indices], y_train[train_indices]
    ##################################################################
    
    print(y_val.std())
    
    
    # Extract settings
    ccd, cn, cmd, mn, md, learning_rate, batch_size = hyper_settings[0], hyper_settings[1], hyper_settings[2], hyper_settings[3], hyper_settings[4], hyper_settings[5], hyper_settings[6]
    EPOCHS = 50
    
    ##################################################################
    ################ CONVOLUTIONAL NEURAL NETWORK ####################
    ##################################################################
    # Create the CNN
    cnn = keras.models.Sequential()

    # Add the input layer
    cnn.add(keras.Input(shape = x_train_image[0].shape)) # Add the input layer
    cnn.add(keras.layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding = 'same', strides = 2))

    # Flatten the final feature layer
    cnn.add(keras.layers.Flatten())
    
    # Add a fully connected dense layer before moving to the output layer
    cnn.add(keras.layers.Dense(cn, activation = "relu"))
    cnn.add(keras.layers.Dropout(cmd))
    cnn.add(keras.layers.Dense(cn, activation = "relu"))
    cnn.add(keras.layers.Dropout(cmd))
    
    # Add the output layer
    cnn.add(keras.layers.Dense(x_train_info.shape[1], activation = "relu")) #TODO

    # give summary of model
    # cnn.summary()
    ##################################################################
    
    ##################################################################
    ################## ARTIFICIAL NEURAL NETWORK #####################
    ##################################################################
    # Create the ANN
    ann = keras.Sequential() 

    # Add the input layer
    ann.add(keras.Input(shape = x_train_info.shape[1])) # Add the input layer

    # give summary of model
    # ann.summary()
    ##################################################################
    
    ##################################################################
    ################# CONCATENATED NEURAL NETWORK ####################
    ##################################################################
    # Combine the two networks (input layer)
    model_concat = keras.layers.concatenate([cnn.output, ann.output])
    model_concat = keras.layers.Dense(mn, activation='relu')(model_concat)
    model_concat = keras.layers.Dropout(md)(model_concat)
    
    # Add the output layer
    model_concat = keras.layers.Dense(1, activation='relu')(model_concat)
    
    # Define the model
    model = keras.models.Model(inputs=[cnn.input, ann.input], outputs=model_concat)
    
    # give summary of model
    model.summary()
    ##################################################################
    
    # Define your sample weights
    sample_weight = np.ones(y_train.shape[0])
    #sample_weight[(y_train) > 40] = 1.2
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = ["mse", keras.metrics.RootMeanSquaredError(name='rmse'), 'mean_absolute_error', keras.losses.Huber(delta=HUBER_DELTA,name='huber_loss')]
    if LOSS == 'huber':
        model.compile(loss=keras.losses.Huber(delta=HUBER_DELTA, name='huber_loss'), optimizer=optimizer, metrics = metrics)
    elif LOSS == 'mse':
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics = metrics)
    elif LOSS == 'mae':
        model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics = metrics)
        

    # Fit the data to the model
    #es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    es = EarlyStopping(monitor='val_rmse', mode='min', verbose=0, patience=5, baseline=y_val.std())
    if save_model: 
        mc = keras.callbacks.ModelCheckpoint("Results/"+CASENAME+"/temp_best_models/"+save_name+'.h5', monitor='val_mse', mode='min', verbose=0, save_best_only=True)
        history = model.fit(x=[x_train_image, x_train_info], y=y_train, batch_size=batch_size, sample_weight=sample_weight, epochs=EPOCHS, validation_data=([x_val_image, x_val_info], y_val), callbacks = [es, mc], verbose = 1)
    else: history = model.fit(x=[x_train_image, x_train_info], y=y_train, batch_size=batch_size, sample_weight=sample_weight, epochs=EPOCHS, validation_data=([x_val_image, x_val_info], y_val), callbacks = [es], verbose = 1)
    
    # Perfromance measures / figures
    # """ ---------- Settings ---------- """
    # PLOT_LOSS_CURVES = False
    # PLOT_FEATURE_MAPS = False
    # if PLOT_FEATURE_MAPS: img = 3
    # """ ------------------------------ """
    
    ##################################################################
    if PLOT_LOSS_CURVES: 
        fig = plt.figure()
        # Loss curves
        plt.plot(pd.DataFrame(history.history)[['loss', 'val_loss']], '-o')
        plt.legend(['Training Loss (MSE)', 'Validation Loss (MSE)'], loc='upper right')
        # Save for latex use
        if not os.path.exists("Results"): os.makedirs("Results")
        pd.DataFrame(history.history)[['loss', 'val_loss']].to_csv('Results/'+CASENAME+'/learning_process_'+str(RANDOM_STATE)+'.csv', sep = ";")
        # Plot the figure 
        plt.show()
    ##################################################################
    
    ##################################################################
    if PLOT_FEATURE_MAPS:
        # summarize feature map shapes
        for i in range(len(cnn.layers)):
            layer = cnn.layers[i]
        	# check for convolutional layer
            if 'conv' not in layer.name:
                continue
        	# summarize output shape
            print(i, layer.name, layer.output.shape)
        
        # redefine model to output right after the first hidden layer
        ixs = [0, 2, 4]
        outputs = [cnn.layers[i+1].output for i in ixs]
        model_short = keras.models.Model(inputs=cnn.inputs, outputs=outputs)
        model_short.summary()
        def plotImages(images, labels = None):
            """
             This procedure can be used to visualize a specific satellite image,
             using the df_satellite dataset (that contains the images themselves)
             a maximum of 40 images can be plotted at the same time
            """
            # Plot max 40 images
            number_figures = len(images)
    
            # Determine widht,height
            max_d = 5
            w = int(min(number_figures, max_d))
            h = int(1 + np.floor((number_figures-1) / max_d))
            try: a = images.index[0]
            except: a = 0
            plt.figure()
            for i in range(0,min(len(images), 40)):
                plt.subplot(h,w,i+1)
                plt.xticks([])
                plt.yticks([])
                if labels is not None:
                    plt.title(str(labels[i]))#df_satellite.highway[i])
                plt.grid(False)
                maxValue = np.amax(images[i+a])
                minValue = np.amin(images[i+a])
                plt.imshow(images[i+a])
            plt.show()
        
        
        
        
        
        for t in range(img, img+1):
            # get feature maps
            
            plotImages(np.expand_dims(x_train_image[t], axis=0))
            if len(ixs)==1: feature_maps = [model_short.predict(np.expand_dims(x_train_image[t], axis=0))]
            else: feature_maps = model_short.predict(np.expand_dims(x_train_image[t], axis=0))

            # plot the output from each block
            count = 0 
            for fmap in feature_maps:
                # plot all 64 maps in an 8x8 squares
                ix = 1
                if count==0: square = 5
                else: square = 8
                fig = plt.figure(figsize=(8,8))
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        ax = plt.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
                        ix += 1

                # show the figure
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
                #fig.tight_layout()
                plt.show()
                count += 1
    ##################################################################
    
    # Return the validation MSE and the amount of epochs
    val_mses = history.history['val_mse']
    val_mse = val_mses[-1]
    val_rmses = history.history['val_rmse']
    val_rmse = val_rmses[-1]
    val_maes = history.history['val_mean_absolute_error']
    val_mae = val_maes[-1]
    val_hubers = history.history['val_huber_loss']
    val_huber = val_hubers[-1]
    n_epochs = len(val_mses)
    return val_mse, val_rmse, val_mae, val_huber, n_epochs

def mixed_neural_network_SPEED_onlySAT(x_train_info, x_train_image, y_train, group_indices, hyper_settings, random_state, save_model = True, save_name = "best_model_SAT"):
    """
     Here we train a deep learning model to predict y_train, based on
     tabular data.

    """
    if not os.path.exists("Results/"+CASENAME+"/temp_best_models"): os.makedirs("Results/"+CASENAME+"/temp_best_models")
    
    ##################################################################
    ##################### CREATE VALIDATION SET ######################
    ##################################################################
    # VALIDATION_SPLIT = 0.2

    # Half of the validation set consists of known roads and half of it of unknown roads
    number_of_samples=len(x_train_info)
    indices_roads = list(group_indices)
    random.Random(random_state).shuffle(indices_roads)
    indices_UR_VS = [item for sublist in indices_roads for item in sublist][0:int(VALIDATION_SPLIT*number_of_samples)] # Unknown roads
    #indices_KR_VS = list(y_train[~y_train.index.isin(indices_UR_VS)].sample(frac=0.5*VALIDATION_SPLIT, replace=False, random_state=random_state).index) # Known roads
    
    # Create validation and training data
    validation_indices = indices_UR_VS##indices_KR_VS + indices_UR_VS#
    #num_known = len(indices_KR_VS)
    train_indices = [e for i, e in enumerate(y_train.index) if i not in validation_indices]
    x_val_info, x_val_image, y_val = x_train_info[validation_indices], x_train_image[validation_indices], y_train[validation_indices]
    x_train_info, x_train_image, y_train = x_train_info[train_indices], x_train_image[train_indices], y_train[train_indices]
    ##################################################################

    
    # Extract settings
    ccd, cn, cmd, mn, md, learning_rate, batch_size = hyper_settings[0], hyper_settings[1], hyper_settings[2], hyper_settings[3], hyper_settings[4], hyper_settings[5], hyper_settings[6]
    EPOCHS = 50

    ##################################################################
    ################ CONVOLUTIONAL NEURAL NETWORK ####################
    ##################################################################
    # Create the CNN
    cnn = keras.models.Sequential()

    # Add the input layer
    cnn.add(keras.Input(shape = x_train_image[0].shape)) # Add the input layer
    cnn.add(keras.layers.Conv2D(32, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.Conv2D(64, kernel_size = (3, 3), padding = 'same', activation='relu')) #TODO
    cnn.add(keras.layers.Dropout(ccd))
    cnn.add(keras.layers.MaxPooling2D(pool_size=(3, 3), padding = 'same', strides = 2))

    # Flatten the final feature layer
    cnn.add(keras.layers.Flatten())
    
    # Add a fully connected dense layer before moving to the output layer
    cnn.add(keras.layers.Dense(cn, activation = "relu"))
    cnn.add(keras.layers.Dropout(cmd))
    cnn.add(keras.layers.Dense(cn, activation = "relu"))
    cnn.add(keras.layers.Dropout(cmd))
    
    # Add the output layer
    cnn.add(keras.layers.Dense(1, activation = "relu")) #TODO

    # give summary of model
    cnn.summary()
    #################################################################
    
    # Define your sample weights
    sample_weight = np.ones(y_train.shape[0])
    
    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = ["mse", keras.metrics.RootMeanSquaredError(name='rmse'), 'mean_absolute_error', keras.losses.Huber(delta=HUBER_DELTA,name='huber_loss')]
    if LOSS == 'huber':
        cnn.compile(loss=keras.losses.Huber(delta=HUBER_DELTA, name='huber_loss'), optimizer=optimizer, metrics = metrics)
    elif LOSS == 'mse':
        cnn.compile(loss='mean_squared_error', optimizer=optimizer, metrics = metrics)
    elif LOSS == 'mae':
        cnn.compile(loss='mean_absolute_error', optimizer=optimizer, metrics = metrics)
        
        
        
    # Fit the data to the model
    #es = keras.callbacks.EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=5)
    es = EarlyStopping(monitor='val_mse', mode='min', verbose=0, patience=5, baseline=y_val.std())
    if save_model:
        mc = keras.callbacks.ModelCheckpoint("Results/"+CASENAME+"/temp_best_models/"+save_name+'.h5', monitor='val_loss', mode='min', verbose=0, save_best_only=True)
        history = cnn.fit(x=x_train_image, y=y_train, batch_size=batch_size, sample_weight=sample_weight, epochs=EPOCHS, validation_data=(x_val_image, y_val), callbacks = [es, mc], verbose = 1)
    else: history = cnn.fit(x=x_train_image, y=y_train, batch_size=batch_size, sample_weight=sample_weight, epochs=EPOCHS, validation_data=(x_val_image, y_val), callbacks = [es], verbose = 1)
    
        
        
        
    #
    #
    #     history = cnn.fit(x=x_train_info, y=y_train, batch_size=batch_size, epochs=EPOCHS, validation_data=(x_val_info, y_val), callbacks = [es, mc], verbose = 1)
    # else: history = cnn.fit(x=x_train_info, y=y_train, batch_size=batch_size, epochs=EPOCHS, validation_data=(x_val_info, y_val), callbacks = [es], verbose = 1)


    # # Perfromance measures / figures
    # """ ---------- Settings ---------- """
    # PLOT_LOSS_CURVES = False
    # """ ------------------------------ """

    ##################################################################
    if PLOT_LOSS_CURVES:
        fig = plt.figure()
        # Loss curves
        plt.plot(pd.DataFrame(history.history)[['loss', 'val_loss']], '-o')
        plt.legend(['Training Loss (MSE)', 'Validation Loss (MSE)'], loc='upper right')
        # Save for latex use
        if not os.path.exists("Results"): os.makedirs("Results")
        pd.DataFrame(history.history)[['loss', 'val_loss']].to_csv('Results/learning_process.csv', sep = ";")
        # Plot the figure
        plt.show()
    ##################################################################

    # Return the validation MSE and the amount of epochs
    # Return the validation MSE and the amount of epochs
    val_mses = history.history['val_mse']
    val_mse = val_mses[-1]
    val_rmses = history.history['val_rmse']
    val_rmse = val_rmses[-1]
    val_maes = history.history['val_mean_absolute_error']
    val_mae = val_maes[-1]
    val_hubers = history.history['val_huber_loss']
    val_huber = val_hubers[-1]
    n_epochs = len(val_mses)
    return val_mse, val_rmse, val_mae, val_huber, n_epochs

    # val_mses = history.history['val_mse']
    # val_mse = val_mses[-1]
    # n_epochs = len(val_mses)
    # return val_mse, n_epochs

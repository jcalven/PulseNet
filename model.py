from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os


class PulseNet:
    """
    PulseNet is a Causal Convolutional Nerual Network for classiyfing raw waveforms, 
    and is built upon the network architecture of WaveNet. The output from PulseNet is a 
    probability tensor of class labels obtained from a SoftMax final layer.
    """

    def __init__(self, data_X, data_y, learning_rate=0.001, n_classes=2, hparams=None,
                 run_dir='', experiment_name='default_experiment_name', is_training=True,
                 causal=True):
        """
        Creates model network on class init.
        Args:
            data_X (tf.Tensor): Input feature tensor.
            data_y (tf.Tensor): Input target tensor.
            learning_rate (float): Learning rate parameter.
            n_classes (int): Number of classes to predict.
            hparams (dict): Dictionary of hyper paramaters.
            run_dir (str): Directory to store model.
            experiment_name (str): Name of experiment.
            is_training (bool): If training, True. 
            causal (bool): If True, dilated conv layer will be causal (as per the WaveNet paper)
        """
    
        self.data_X = data_X
        self.data_y = data_y
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.hparams = hparams
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.is_training = is_training
        self.causal = causal
        self._create_architecture(data_X, data_y, run_dir, experiment_name)
        
    def _create_architecture(self, data_X, data_y, run_dir, experiment_name):

        logits = self._create_model(data_X)
        
        # Average pooling layer (directly after the dilated conv layer, as per the WaveNet paper)
        logits = self.average_pooling(logits, pool_size=self.hparams.pool_size, avg_pool_padding=self.hparams.avg_pool_padding)
        
        # Add here one or more conv_1x1_layer (as per the WaveNet paper)
        logits = self.conv_1x1_layer(logits, logits.shape[-1].value)
        
        with tf.variable_scope("postprocess"):
            predictions = tf.keras.activations.relu(logits)     
            predictions = self.conv_1x1_layer(
                predictions,
                output_size=predictions.shape[-1].value,
                activation="relu",
            )
              
        # Batch norm
        predictions = tf.keras.layers.BatchNormalization()(predictions, training=self.is_training)
            
        predictions = tf.layers.flatten(predictions) 
        predictions = tf.keras.layers.Dense(self.n_classes, activation="softmax")(predictions)   
        
        # Assign class variable for predicted labels for easy retrieval
        self.y_predict = predictions
        
        # Use softmax cross entropy since labels are one hot encoded
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = data_y, logits = predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
        
        decoded_predictions = tf.argmax(predictions, axis=1)
        decoded_data_y = tf.argmax(data_y, axis=1)
        
        self.correct_prediction = tf.equal(tf.argmax(data_y, 1), tf.argmax(predictions, 1))
        self.accuracy, self.accuracy_update_op = tf.metrics.accuracy(tf.argmax(data_y, 1), tf.argmax(predictions, 1), name='accuracy')
        self.auc, self.auc_update_op = tf.metrics.auc(data_y, predictions, name='auc', num_thresholds=5000, summation_method='careful_interpolation')
        
        # Create a summary to monitor cost (loss) tensor
        tf.summary.scalar("loss", self.loss)
        # Create a summary to monitor accuracy tensor
        tf.summary.scalar("accuracy", self.accuracy)
        # Create a summary to monitor ROC AUC tensors
        tf.summary.scalar("auc", self.auc)
        tf.summary.scalar("auc_update_op", self.auc_update_op)
        # Merge all summaries into a single op
        self.summaries = tf.summary.merge_all()
        
        # Check and set save dir
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        if not os.path.exists(os.path.join(run_dir, "checkpoints")):
            os.mkdir(os.path.join(run_dir, "checkpoints"))
        if not os.path.exists(os.path.join(run_dir, "checkpoints", experiment_name)):
            os.mkdir(os.path.join(run_dir, "checkpoints", experiment_name))
        self._run_dir = run_dir
        self.saver = tf.train.Saver(max_to_keep=1)

    def _create_model(self, X):
        """Builds PulseNet network.
        This consists of:
          1) An initial causal convolution,
          2) The dilated causal convolutional stack, and
          3) Summing of skip connections
        The network output can then be used for classification of waveforms.
        
        Args:
            X (tf.Tensor): Feature tensor
        Returns:
            tf.Tensor: Resulting sum of skip connections
        """
        skip_connections = []
        x = X
        with tf.variable_scope("preprocess"):
            x = self.causal_conv_layer(
                x,
                self.hparams.preprocess_output_size,
                self.hparams.preprocess_kernel_width,
            )
        for i in range(self.hparams.num_residual_blocks):
            with tf.variable_scope("block_{}".format(i)):
                for dilation_rate in self.hparams.dilation_rates:
                    with tf.variable_scope("dilation_{}".format(dilation_rate)):
                        skip_connection, x = self.gated_residual_layer(x, dilation_rate)
                        skip_connections.append(skip_connection)

        network_output = tf.add_n(skip_connections)
        return network_output
        
    def causal_conv_layer(self, x, output_size, kernel_width, dilation_rate=1):
        """Applies a dilated causal convolution to the input.
        Args:
            x (tf.Tensor): Input tensor.
            output_size (int): Number of output filters for the convolution.
            kernel_width (int): Width of the 1D convolution window.
            dilation_rate (int): Dilation rate of the layer.
        Returns:
            tf.Tensor: Resulting tensor after applying the convolution.
        """
        if self.causal:
            padding = 'causal'
        else:
            padding = 'same'
        
        causal_conv_op = tf.keras.layers.Conv1D(
            output_size,
            kernel_width,
            padding=padding,
            dilation_rate=dilation_rate,
            name="causal_conv",
        )
        return causal_conv_op(x)

    def conv_1x1_layer(self, x, output_size, activation=None):
        """Applies a 1x1 convolution to the input.
        Args:
            x (tf.Tensor): Input tensor.
            output_size (int): Number of output filters for the 1x1 convolution.
            activation (str): Activation function to apply (e.g. 'relu').
        Returns:
            tf.Tensor: Resulting tensor after applying the 1x1 convolution.
        """
        conv_1x1_op = tf.keras.layers.Conv1D(
            output_size, 1, activation=activation, name="conv1x1"
        )
        return conv_1x1_op(x)
    
    def average_pooling(self, x, pool_size=2, avg_pool_padding='same'):
        """Applies a 1D average pooling operation to the input.
        Args:
            x (tf.Tensor): Input tensor.
        Returns:
            tf.Tensor: Resulting tensor after applying average pooling.
        """
        avg_pool = tf.keras.layers.AveragePooling1D(pool_size=pool_size, padding=avg_pool_padding)
        return avg_pool(x)

    def gated_residual_layer(self, x, dilation_rate):
        """Creates a gated, dilated convolutional layer with a residual connnection.
        Args:
            x (tf.Tensor): Input tensor
            dilation_rate (int): Dilation rate of the layer.
        Returns:
            tf.Tensor: Skip connection to network_output layer.
            tf.Tensor: Sum of learned residual and input tensor.
        """
        with tf.variable_scope("filter"):
            x_filter_conv = self.causal_conv_layer(
                x, x.shape[-1].value, self.hparams.dilation_kernel_width, dilation_rate
            )
        with tf.variable_scope("gate"):
            x_gate_conv = self.causal_conv_layer(
                x, x.shape[-1].value, self.hparams.dilation_kernel_width, dilation_rate
            )
            
        gated_activation = tf.tanh(x_filter_conv) * tf.sigmoid(x_gate_conv)

        with tf.variable_scope("residual"):
            residual = self.conv_1x1_layer(gated_activation, x.shape[-1].value)
        with tf.variable_scope("skip"):
            skip_connection = self.conv_1x1_layer(
                gated_activation, self.hparams.skip_output_dim
            )
        return skip_connection, x + residual
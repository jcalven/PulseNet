import os, sys
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import json

import warnings
warnings.simplefilter("ignore", category=(PendingDeprecationWarning, DeprecationWarning))

# Import wavenetclass modules
from config import ConfigDict, config_reader
from model import PulseNet
from dataset import PreProcess


class Train(object):
    
    def __init__(self, config_file, modes=None, index_key='id'):
        self.config_file = config_file
        self.modes = modes
        self.index_key = index_key
        # Initialize config 
        self.config = self.get_config()

    def get_config(self):
        """
        Creates a ConfigDict with all config file entries. Adds shortcuts to certain entries.
        Args:
            config (str): Path to config file.
            modes (str, List(str)): Can be either one of ['train', 'val', 'test'] or list of one or all of them.
            index_key (str): Field to use for comparing index values.
        Returns:

        """        
        out = ConfigDict()
        out['LOGS'] = config_reader(self.config_file, KEYS='PATHS', entry='LOGS').get('PATHS')
        out['DATA'] = config_reader(self.config_file, KEYS=self.modes, entry='files')
        out['NAME'] = config_reader(self.config_file, KEYS='EXPERIMENT', entry='name').get('EXPERIMENT')

        # Load index
        index_files = config_reader(self.config_file, KEYS=self.modes, entry='index')
        index = {}
        for mode, index_file in index_files.items():
            with open(index_file, 'r') as fp:
                index[mode] = {self.index_key: json.load(fp).get(mode.lower())}
        out['INDEX'] = ConfigDict(**index)

        # Retrieve all major keys in config file
        out['PATHS'] = config_reader(self.config_file, KEYS='PATHS').get('PATHS')
        out['HPARAMS'] = config_reader(self.config_file, KEYS='HPARAMS').get('HPARAMS')
        out['PIPELINE'] = config_reader(self.config_file, KEYS='PIPELINE').get('PIPELINE')
        out['EXPERIMENT'] = config_reader(self.config_file, KEYS='EXPERIMENT').get('EXPERIMENT')
        out['TRAIN'] = config_reader(self.config_file, KEYS='TRAIN').get('TRAIN')
        out['VAL'] = config_reader(self.config_file, KEYS='VAL').get('VAL')
        out['TEST'] = config_reader(self.config_file, KEYS='TEST').get('TEST')
        return out

    def print_info(self):
        print('###################################')
        print('          PulseNet v0.1            ')
        print('\nPipeline settings:')
        for key, val in self.config.PIPELINE.items():
            print(key, '=', val)
        print('\nHyper parameters:')
        for key, val in self.config.HPARAMS.items():
            print(key, '=', val)
        print('\nExperiment name:', self.config.NAME)
        print('\nStarting training...')
        print('\nUsing training data <{}>'.format(self.config.DATA.TRAIN))
        print('\nUsing validation data <{}>'.format(self.config.DATA.VAL))
        print()

    def run(self):
        """Main method for training PulseNet model."""

        # Assign config variable to local variable
        config = self.config
        
        preprocess_train = PreProcess(index=config.INDEX.TRAIN, **config.PIPELINE)
        preprocess_val = PreProcess(index=config.INDEX.VAL, **config.PIPELINE)

        # Prep train and eval data
        train_dataset = preprocess_train.prep(config.DATA.TRAIN)
        val_dataset = preprocess_val.prep(config.DATA.VAL)

        # Dataset iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
        train_data_X, train_data_y = train_dataset.make_one_shot_iterator().get_next()
        val_data_X, val_data_y = val_dataset.make_one_shot_iterator().get_next()
        data_X, data_y = iterator.get_next()

        # Initialize with required Datasets
        train_iterator = iterator.make_initializer(train_dataset)
        val_iterator = iterator.make_initializer(val_dataset)

        # Length of wavelength
        X_length = preprocess_train.stop_index - preprocess_train.start_index 

        placeholder_X = tf.placeholder(tf.float32, [None, X_length, 1])
        placeholder_y = tf.placeholder(tf.int64, [None, 2])

        # Custom summaries
        train_epoch_acc_summary = tf.Summary()
        train_epoch_auc_summary = tf.Summary()
        val_epoch_acc_summary = tf.Summary()
        val_epoch_auc_summary = tf.Summary()

        loss_arr = []
        acc_arr = []
        epoch_arr = []

        time_string = datetime.datetime.now().isoformat()
        config.NAME += config.NAME + '_{}'.format(time_string)

        # Instantiate model
        model = PulseNet(data_X, data_y, hparams=config.HPARAMS, run_dir=config.LOGS, learning_rate=config.HPARAMS.learning_rate, 
                         experiment_name=config.NAME, causal=True)

        # Prints info before training starts
        self.print_info()

        # Store loss
        hist_loss = []

        # Run session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Write logs to Tensorboard
            train_writer = tf.summary.FileWriter(config.LOGS+'train/'+config.NAME, sess.graph)
            val_writer = tf.summary.FileWriter(config.LOGS+'val/'+config.NAME, sess.graph)

            n_train_samples = 0
            n_val_samples = 0

            for epoch_no in range(config.HPARAMS.epochs):

                train_loss, train_accuracy = 0, 0
                val_loss, val_accuracy = 0, 0

                X_train, y_train = sess.run((train_data_X, train_data_y))
                X_val, y_val = sess.run((val_data_X, val_data_y))

                # Initialize iterator with training data
                sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})

                # Set model to training mode
                model.is_training = True

                i_batch = 0
                try:
                    with tqdm(total = len(y_train)) as pbar:
                        while i_batch <= config.HPARAMS.n_train_batches:
                            _, loss, acc_update_op, summary = sess.run([model.optimizer, model.loss, model.accuracy_update_op,
                                                     model.summaries])
                            train_loss += loss 
                            n_train_samples += config.PIPELINE.batch
                            pbar.update(config.PIPELINE.batch)

                            if i_batch % config.PIPELINE.train_sample_rate == 0:
                                print('\nEpoch {}: batch = {}, loss = {:.4f}, accuracy = {:.4f}'.format(epoch_no, i_batch+1, train_loss/(i_batch+1), acc_update_op))
                                # Write logs at every iteration
                                train_writer.add_summary(summary, n_train_samples)

                            i_batch += 1
                except tf.errors.OutOfRangeError:
                    print('End of dataset')

                # After every epoch, calculate the accuracy of the last seen training batch 
                acc_train = sess.run(model.accuracy)

                # Add logs at end of train epoch
                train_epoch_acc_summary.value.add(tag="epoch_accuracy", simple_value=acc_train)
                train_writer.add_summary(train_epoch_acc_summary, epoch_no)

                print("Epoch {}: training loss = {:.3f}, training accuracy: {:.2f}%".format(epoch_no, train_loss/(i_batch), acc_train * 100))

                # Early stopping
                hist_loss.append(train_loss/(i_batch))
                patience_cnt = 0
                if epoch_no > 0:
                    if hist_loss[epoch_no-1] - hist_loss[epoch_no] > config.HPARAMS.min_delta:
                        patience_cnt = 0
                    else:
                        patience_cnt += 1

                if patience_cnt > config.HPARAMS.patience:
                    print("\nEarly stopping...")
                    # Save model
                    print('\nSaving model...')
                    model.saver.save(sess, 'logs/checkpoints/'+config.NAME+'/model')
                    print('ok\n')
                    break

                # Save model
                print('\nSaving model...')
                model.saver.save(sess, 'logs/checkpoints/'+config.NAME+'/model')
                print('ok\n')

                # Initialize iterator with validation data
                sess.run(val_iterator, feed_dict = {placeholder_X: X_val, placeholder_y: y_val})

                # Set model to validation mode
                model.is_training = False

                i_batch = 1
                try:
                    with tqdm(total = len(y_val)) as pbar:
                        while i_batch <= config.HPARAMS.n_val_batches:
                            loss, val_acc_update_op, summary, auc, auc_update_op = sess.run([model.loss, model.accuracy_update_op,
                                                     model.summaries, model.auc, model.auc_update_op])
                            val_loss += loss
                            n_val_samples += config.PIPELINE.batch
                            pbar.update(config.PIPELINE.batch)

                            # Write logs at every iteration
                            val_writer.add_summary(summary, n_val_samples)

                            i_batch += 1
                except tf.errors.OutOfRangeError:
                    print('End of dataset')

                # After each epoch, calculate the accuracy of the test data
                acc_val, auc_val = sess.run([model.accuracy, model.auc])
                print('AUC =', auc_val)

                auc_local_variables = [str(i.name) for i in tf.local_variables() if str(i.name).startswith('auc')]
                roc_dict = {vn.split('/')[-1].split(':')[0]: sess.run(tf.get_default_graph().get_tensor_by_name(vn)) for vn in auc_local_variables}

                # Add logs at end of train epoch
                val_epoch_acc_summary.value.add(tag="epoch_accuracy", simple_value=acc_val)
                val_epoch_auc_summary.value.add(tag="epoch_AUC", simple_value=auc_val)
                val_writer.add_summary(val_epoch_acc_summary, epoch_no)
                val_writer.add_summary(val_epoch_auc_summary, epoch_no)

                print("Average validation set accuracy and loss over {} batch iterations are {:.2f}% and {:.2f}".format(i_batch, acc_val * 100, val_loss / i_batch))

            # Calculate and save True Positive Rate (TPR) and False Positive Rate (FPR)
            tpr = roc_dict['true_positives'] / (roc_dict['true_positives'] + roc_dict['false_negatives'])
            tnr = roc_dict['true_negatives'] / (roc_dict['true_negatives'] + roc_dict['false_positives'])
            fpr = 1 - tnr

            roc_dict['auc'] = auc_val
            roc_dict['tpr'] = tpr
            roc_dict['tnr'] = fpr
            roc_dict['fpr'] = fpr

            log_data_path = config.LOGS + 'checkpoints' + '/' + config.NAME + '/'

             # Dump AUC ROC data to json
            with open(log_data_path + 'roc_auc.json', 'w') as fp:
                roc_dict_as_list = {key: val.tolist() for key, val in roc_dict.items()}
                json.dump(roc_dict_as_list, fp)

            # Dump configuration to json
            with open(log_data_path + 'config.json', 'w') as fp:
                json.dump({'hparams': config.HPARAMS, 'pipeline_settings': config.PIPELINE}, fp)

            train_writer.close()
            val_writer.close()

            print('\nTraining finished.')
            
            
if __name__ == '__main__':
    
    tf.reset_default_graph()   # Clears the defined variables and operations of the previous cell
    
    # Run
    config = 'default.ini'
    modes = ['train', 'val']
    train = Train(config_file=config, modes=modes)
    train.run()
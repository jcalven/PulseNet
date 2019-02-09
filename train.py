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
from configdict import ConfigDict
from model import PulseNet
from dataset import PreProcess

def run(train_files, val_files, hparams, pipeline_settings, logs_path='./logs/', experiment_name='PulseNet_default'):
    """Main method for training PulseNet model."""
    
    preprocess = PreProcess(**pipeline_settings)
    
    # Prep train and eval data
    train_dataset = preprocess.prep(train_files)
    val_dataset = preprocess.prep(val_files)

    # Dataset iterator
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_data_X, train_data_y = train_dataset.make_one_shot_iterator().get_next()
    val_data_X, val_data_y = val_dataset.make_one_shot_iterator().get_next()
    data_X, data_y = iterator.get_next()

    # Initialize with required Datasets
    train_iterator = iterator.make_initializer(train_dataset)
    val_iterator = iterator.make_initializer(val_dataset)

    # Length of wavelength
    X_length = preprocess.stop_index - preprocess.start_index 
    
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
    experiment_name += '_{}'.format(time_string)

    # Instantiate model
    model = PulseNet(data_X, data_y, hparams=hparams, run_dir=logs_path, learning_rate=hparams.learning_rate, experiment_name=experiment_name)
    
    print('Experiment name:', experiment_name, '\n')
    print('Starting training...\n')
    
    # Run session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Write logs to Tensorboard
        train_writer = tf.summary.FileWriter(logs_path+'train/'+experiment_name, sess.graph)
        val_writer = tf.summary.FileWriter(logs_path+'val/'+experiment_name, sess.graph)
        
        n_train_samples = 0
        n_val_samples = 0

        for epoch_no in range(hparams.EPOCHS):

            train_loss, train_accuracy = 0, 0
            val_loss, val_accuracy = 0, 0

            X_train, y_train = sess.run((train_data_X, train_data_y))
            X_val, y_val = sess.run((val_data_X, val_data_y))

            # Initialize iterator with training data
            sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})

            i_batch = 0
            try:
                with tqdm(total = len(y_train)) as pbar:
                    while i_batch <= hparams.n_train_batches:
                        _, loss, acc_update_op, summary = sess.run([model.optimizer, model.loss, model.accuracy_update_op,
                                                 model.summaries])
                        train_loss += loss 
                        n_train_samples += pipeline_settings['batch']
                        pbar.update(pipeline_settings['batch'])
                        
                        if i_batch % pipeline_settings.train_sample_rate == 0:
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

            print("Epoch: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(epoch_no, loss, acc_train * 100))

            # Save model
            print('\nSaving model...')
            model.saver.save(sess, 'logs/checkpoints/'+experiment_name+'/model')
            print('ok\n')

            # Initialize iterator with validation data
            sess.run(val_iterator, feed_dict = {placeholder_X: X_val, placeholder_y: y_val})

            i_batch = 1
            try:
                with tqdm(total = len(y_val)) as pbar:
                    while i_batch <= hparams.n_train_batches:
                        loss, val_acc_update_op, summary, auc, auc_update_op = sess.run([model.loss, model.accuracy_update_op,
                                                 model.summaries, model.auc, model.auc_update_op])
                        val_loss += loss
                        n_val_samples += pipeline_settings['batch']
                        pbar.update(pipeline_settings['batch'])

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
        
        log_data_path = logs_path + 'checkpoints' + '/' + experiment_name + '/'
        
         # Dump AUC ROC data to json
        with open(log_data_path + 'roc_auc.json', 'w') as fp:
            roc_dict_as_list = {key: val.tolist() for key, val in roc_dict.items()}
            json.dump(roc_dict_as_list, fp)
        
        # Dump configuration to json
        with open(log_data_path + 'config.json', 'w') as fp:
            json.dump({'hparams': hparams, 'pipeline_settings': pipeline_settings}, fp)

        train_writer.close()
        val_writer.close()
        
        print('\nTraining finished.')
        

def run_main(experiment_name, pipeline_settings=None, hparams=None):
    
    # Default input pipeline settings ###
    pipeline_settings_run = ConfigDict(
        buffer_size = 1, #100,
        num_parallel_calls = 1, #20,
        prefetch = 1, #10,
        cycle_length = 1, #2,
        repeat = 1,
        batch=1, #128,
        stop_index=10,
        train_sample_rate=10)
    
    # Update pipeline_settings_run parameters
    if isinstance(hparams, dict):
        for key, val in pipeline_settings.items():
            if key in pipeline_settings_run:
                pipeline_settings_run.update({key:val})

    # Default hyper parameters
    hparams_run = ConfigDict(
        dilation_kernel_width = 4, #2
        skip_output_dim = 2, #1
        preprocess_output_size = 1,
        preprocess_kernel_width = 4, #1
        num_residual_blocks = 3,
        dilation_rates = [1, 2, 4],
        EPOCHS = 3,
        n_train_batches=500,
        n_val_batches=500,
        learning_rate=0.002)#0.001)
    
    # Update hparams_run parameters
    if isinstance(hparams, dict):
        for key, val in hparams.items():
            if key in hparams_run:
                hparams_run.update({key:val})
            
    print('###################################')
    print('          PulseNet v0.1            ')
    print('\nPipeline settings:')
    for key, val in pipeline_settings_run.items():
        print(key, '=', val)
    print('\nHyper parameters:')
    for key, val in hparams_run.items():
        print(key, '=', val)
    print()        

    # Paths
    SRC_PATH = '/cfs/klemming/nobackup/j/jcalven/lar/notebooks/wavenetclass/'
    DATA_PATH = SRC_PATH+'data/'
    LOGS_PATH = SRC_PATH+'version_0_1/logs/'
    
    
    # Data files
    train_files = ['path/to/train_data.h5']
    val_files = ['path/to/val_data.h5']

    
    # Run training
    roc_dict = run(train_files=train_files, val_files=val_files, hparams=hparams_run, 
        pipeline_settings=pipeline_settings_run, logs_path=LOGS_PATH, experiment_name=experiment_name)
    
    return roc_dict
    
    
if __name__ == '__main__':
        
    experiment_name = 'test_clean'
    
    roc_dict = run_main(experiment_name, pipeline_settings=None, hparams=None)
    
    plot = False
    
    if plot:
        tpr = roc_dict['true_positives'] / (roc_dict['true_positives'] + roc_dict['false_negatives'])
        tnr = roc_dict['true_negatives'] / (roc_dict['true_negatives'] + roc_dict['false_positives'])
        fpr = 1 - tnr

        with plt.style.context(('bmh')):
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6,6))
            ax.plot(fpr, tpr, label='1 epoch')
            ax.plot([0,1], [0,1], '--')
            ax.text(x=0.81, y=0.21, s='AUC = {:0.2}'.format(roc_dict['auc']))
            ax.set_xlim(0.,1.)
            ax.set_ylim(0.,1.)
            ax.set_title('ROC')
            ax.set_xlabel('False Positive Rate (1 - specificity)')
            ax.set_ylabel('True Positive Rate (sensitivity)')
            fig.savefig(experiment_name +'.pdf', bbox_inches='tight')
            fig.savefig(experiment_name +'.png', dpi=300, bbox_inches='tight')
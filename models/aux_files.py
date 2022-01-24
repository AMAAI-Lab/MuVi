#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import torch
import numpy as np
from audtorch.metrics.functional import concordance_cc
from scipy import stats
import sklearn
import pandas as pd
from tensorflow.keras import models,layers, optimizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


stop_train = EarlyStopping( monitor='val_loss', patience=7, verbose=0, mode='min')


def print_results(results):
    
    #for valence
    val = results.loc[results['emotion'] == 'valence'].to_numpy()
    val_mse_mean = np.mean(val[:,2])
    val_mse_std = np.std(val[:,2])
    val_ccc_mean = np.mean(val[:,3])
    val_ccc_std = np.std(val[:,3])
    print('For Valence')
    print('MSE (avg) is: ', val_mse_mean, 'with deviation ', val_mse_std)
    print('CCC (avg) is: ', val_ccc_mean, 'with deviation ', val_ccc_std)
    
    aro = results.loc[results['emotion'] == 'arousal'].to_numpy()
    aro_mse_mean = np.mean(aro[:,2])
    aro_mse_std = np.std(aro[:,2])
    aro_ccc_mean = np.mean(aro[:,3])
    aro_ccc_std = np.std(aro[:,3])
    print('For Arousal')
    print('MSE (avg) is: ', aro_mse_mean, 'with deviation ', aro_mse_std)
    print('CCC (avg) is: ', aro_ccc_mean, 'with deviation ', aro_ccc_std)
    

def train_lstm(hp, opensmileFeatureDir, directory_video, ewe, modality, urls, av_df, allsplits):
    
    #get the right training process
    if modality == 'music':
        results = music_modality_train(hp, opensmileFeatureDir, ewe, modality, urls, av_df, allsplits)
    elif modality == 'muted_video':
        results = muted_video_modality_train(hp, directory_video, ewe, modality, urls, av_df, allsplits)
    elif modality == 'video_muted':
        results = video_muted_modality_train(hp, directory_video, ewe, modality, urls, av_df, allsplits)
    elif modality == 'video_music':
        results = video_music_modality_train(hp, opensmileFeatureDir, ewe, modality, urls, av_df, allsplits)
    elif modality == 'musicvideoConcat':
        results = music_video_modality_concat(hp, opensmileFeatureDir, directory_video, ewe, modality, urls, av_df, allsplits)
    elif modality == 'musicvideoSep':
        results = music_video_modality_sep(hp, opensmileFeatureDir, directory_video, ewe, modality, urls, av_df, allsplits)
    else: #musicvideoShared
        results = music_video_modality_shared(hp, opensmileFeatureDir, directory_video, ewe, modality, urls, av_df, allsplits)
    
    return results


def music_video_modality_shared(hp, opensmileFeatureDir, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        
        #models loading
        aro_audio_model = models.load_model('./models/music_aro_LSTM.h5')
        aro_muted_model = models.load_model('./models/muted_aro_LSTM.h5')
        val_audio_model = models.load_model('./models/music_val_LSTM.h5')
        val_muted_model = models.load_model('./models/muted_val_LSTM.h5')        
        
        #Arousal
        X_train_a,  X_train_v, y_arousal, y_valence = keras_load_data_all_sep(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], 
                                                            ewe, opensmileFeatureDir)
        X_val_a, X_val_v, y_val_arousal, y_val_valence = keras_load_data_all_sep(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe, 
                                                                  opensmileFeatureDir)
        X_test_a, X_test_v, y_test_arousal, y_test_valence = keras_load_data_all_sep(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe, 
                                                                     opensmileFeatureDir)
           
        
        X_train_a, X_val_a, X_test_a = scale_lstm_features(X_train_a, X_val_a, X_test_a)
        X_train_v, X_val_v, X_test_v = scale_lstm_features(X_train_v, X_val_v, X_test_v)
        #numOfFes_a = X_train_a.shape[-1]
        #numOfFes_v = X_train_v.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelLoader_LSTM_all(aro_audio_model, aro_muted_model, hp['dense_units'])
       
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit({'audio_features':X_train_a, 'muted_features': X_train_v},
                  {'out_var1': y_arousal }, epochs=100, validation_data=([X_val_a, X_val_v], y_val_arousal),
                  callbacks = stop_train, batch_size=512)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test_a.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc, preds_aro = lstm_scoring_twoInps(model, X_test_a[start_idx:end_idx, ], X_test_v[start_idx:end_idx, ], 
                                                      y_test_arousal[start_idx:end_idx, ])
            #store results
            results.append([i, 'arousal',test_mse, test_ccc, urls.iloc[test_index[j]][0], preds_aro])
        #Valence
     
        #compile and fit model for VALENCE
        model = modelLoader_LSTM_all(val_audio_model, val_muted_model, hp['dense_units'])
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit({'audio_features':X_train_a, 'muted_features': X_train_v},
                {'out_var1': y_valence }, epochs=100, validation_data=([X_val_a, X_val_v], y_val_valence),
                callbacks = stop_train, batch_size=512)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test_a.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc, preds_val = lstm_scoring_twoInps(model, X_test_a[start_idx:end_idx, ], X_test_v[start_idx:end_idx, ],  
                                                      y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc, urls.iloc[test_index[j]][0], preds_val])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc','song_id','preds'])
    
    return results

def music_video_modality_sep(hp, opensmileFeatureDir, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train_a,  X_train_v, y_arousal, y_valence = keras_load_data_all_sep(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], 
                                                            ewe, opensmileFeatureDir)
        X_val_a, X_val_v, y_val_arousal, y_val_valence = keras_load_data_all_sep(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe, 
                                                                  opensmileFeatureDir)
        X_test_a, X_test_v, y_test_arousal, y_test_valence = keras_load_data_all_sep(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe, 
                                                                     opensmileFeatureDir)
           
        
        X_train_a, X_val_a, X_test_a = scale_lstm_features(X_train_a, X_val_a, X_test_a)
        X_train_v, X_val_v, X_test_v = scale_lstm_features(X_train_v, X_val_v, X_test_v)
        numOfFes_a = X_train_a.shape[-1]
        numOfFes_v = X_train_v.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM_sep(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes_a, numOfFes_v)
       
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit({'audio_features':X_train_a, 'muted_features': X_train_v},
                  {'out_var1': y_arousal }, epochs=100, validation_data=([X_val_a, X_val_v], y_val_arousal),
                  callbacks = stop_train, batch_size=512)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test_a.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring_twoInps(model, X_test_a[start_idx:end_idx, ], X_test_v[start_idx:end_idx, ], 
                                                      y_test_arousal[start_idx:end_idx, ])
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
     
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM_sep(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes_a, numOfFes_v)
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit({'audio_features':X_train_a, 'muted_features': X_train_v},
                {'out_var1': y_valence }, epochs=100, validation_data=([X_val_a, X_val_v], y_val_valence),
                callbacks = stop_train, batch_size=512)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test_a.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring_twoInps(model, X_test_a[start_idx:end_idx, ], X_test_v[start_idx:end_idx, ],  
                                                      y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results


def music_video_modality_concat(hp, opensmileFeatureDir, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train, y_arousal, y_valence = keras_load_data_all(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], 
                                                            ewe, opensmileFeatureDir)
        X_val, y_val_arousal, y_val_valence = keras_load_data_all(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe, 
                                                                  opensmileFeatureDir)
        X_test, y_test_arousal, y_test_valence = keras_load_data_all(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe, 
                                                                     opensmileFeatureDir)
    
        X_train, X_val, X_test = scale_lstm_features(X_train, X_val, X_test)
        
        numOfFes = X_train.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, mod)
       
        
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit(X_train, y_arousal, epochs=100, validation_data=(X_val, y_val_arousal),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_arousal[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
    
        
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, mod)

    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit(X_train, y_valence, epochs=100, validation_data=(X_val, y_val_valence),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results


def muted_video_modality_train(hp, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train, y_arousal, y_valence = keras_load_data_video(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], ewe)
        X_val, y_val_arousal, y_val_valence = keras_load_data_video(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe)
        X_test, y_test_arousal, y_test_valence = keras_load_data_video(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe)
    
        X_train, X_val, X_test = scale_lstm_features(X_train, X_val, X_test)
        
        numOfFes = X_train.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
       
        
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
        save_model = ModelCheckpoint('./models/muted_aro_LSTM.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
        model.fit(X_train, y_arousal, epochs=100, validation_data=(X_val, y_val_arousal),
                            callbacks = [stop_train, save_model], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_arousal[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
    
        
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
        save_model = ModelCheckpoint('./models/muted_val_LSTM.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
        model.fit(X_train, y_valence, epochs=100, validation_data=(X_val, y_val_valence),
                            callbacks = [stop_train, save_model], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results


def video_muted_modality_train(hp, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train, y_arousal, y_valence = keras_load_muted_video(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], ewe)
        X_val, y_val_arousal, y_val_valence = keras_load_muted_video(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe)
        X_test, y_test_arousal, y_test_valence = keras_load_muted_video(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe)
    
        X_train, X_val, X_test = scale_lstm_features(X_train, X_val, X_test)
        
        numOfFes = X_train.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
       
        
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit(X_train, y_arousal, epochs=100, validation_data=(X_val, y_val_arousal),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_arousal[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
    
        
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
    
    
        model.fit(X_train, y_valence, epochs=100, validation_data=(X_val, y_val_valence),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results

def video_music_modality_train(hp, directory_video, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train, y_arousal, y_valence = keras_load_music_video(av_df, urls.iloc[train_index], directory_video, hp['seq_len'], ewe)
        X_val, y_val_arousal, y_val_valence = keras_load_music_video(av_df, urls.iloc[val_index], directory_video, hp['seq_len'], ewe)
        X_test, y_test_arousal, y_test_valence = keras_load_music_video(av_df, urls.iloc[test_index], directory_video, hp['seq_len'], ewe)
    
        X_train, X_val, X_test = scale_lstm_features(X_train, X_val, X_test)
        
        numOfFes = X_train.shape[-1]
        
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
       
        
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
    
        model.fit(X_train, y_arousal, epochs=100, validation_data=(X_val, y_val_arousal),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_arousal[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
    
        
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'muted')
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
    
    
        model.fit(X_train, y_valence, epochs=100, validation_data=(X_val, y_val_valence),
                            callbacks = [stop_train], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits
    
    
    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results


def music_modality_train(hp, opensmileFeatureDir, ewe, mod, urls, av_df, allsplits):
    
    results = []
    i=0
    
    for splits in allsplits:
        train_index = splits[0]
        val_index = splits[1]
        test_index = splits[2]
        #Arousal
        X_train, y_arousal, y_valence = keras_load_data_music(av_df, urls.iloc[train_index], opensmileFeatureDir, hp['seq_len'], ewe)
        X_val, y_val_arousal, y_val_valence = keras_load_data_music(av_df, urls.iloc[val_index], opensmileFeatureDir, hp['seq_len'], ewe)
        X_test, y_test_arousal, y_test_valence = keras_load_data_music(av_df, urls.iloc[test_index], opensmileFeatureDir, hp['seq_len'], ewe)
    
        X_train, X_val, X_test = scale_lstm_features(X_train, X_val, X_test)
        
        numOfFes = X_train.shape[-1]
    
        #compile and fit model for AROUSAL
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'audio')
    
        
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        save_model = ModelCheckpoint('./models/music_aro_LSTM.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
        model.fit(X_train, y_arousal, epochs=100, validation_data=(X_val, y_val_arousal),
                            callbacks = [stop_train,save_model], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_arousal[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'arousal',test_mse, test_ccc])
        #Valence
    
    
        #compile and fit model for VALENCE
        model = modelBuilder_LSTM(hp['seq_len'], hp['dense_units'], hp['lstm_units'], numOfFes, 'audio')
    
        
    
        model.compile(
            loss = 'mean_squared_error',
            optimizer = optimizers.Adam(learning_rate=hp['lr']),
            metrics = metrics.RootMeanSquaredError()
        )
        
        save_model = ModelCheckpoint('./models/music_val_LSTM.h5', monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')
    
        model.fit(X_train, y_valence, epochs=100, validation_data=(X_val, y_val_valence),
                            callbacks = [stop_train,save_model], batch_size=256)
    
    
        #get results for each media item in test set (which should be length==111)
        for j in range(int(X_test.shape[0]/111)):
            start_idx, end_idx = j*111, ((j+1)*111)
            test_mse, test_ccc = lstm_scoring(model, X_test[start_idx:end_idx, ], y_test_valence[start_idx:end_idx, ])
    
            #store results
            results.append([i, 'valence',test_mse, test_ccc])
    
        i+=1 #track splits


    results = pd.DataFrame(results, columns=['split', 'emotion','mse','ccc'])
    
    return results


def modelBuilder_LSTM(seq_len, dense_units, lstm_units, numOfFes, mod):
    """
    define LSTM model architecture using Keras' functional API. This one is for emobase features (988 audio features)
    """

    #LSTM with openSMILE audio features as input
    audio_inputs = layers.Input(shape=(seq_len, numOfFes), name=mod+"_features")
    #x = layers.Conv1D(filters=lstm_units, kernel_size=2, activation='relu')(audio_inputs)
    #x = layers.Conv1D(filters=dense_units, kernel_size=2, activation='relu')(x)

    #call a layer on model_inputs, in this case applying Dense layer to each timestep 
    x = layers.TimeDistributed(layers.Dense(dense_units), name=mod+'_dense')(audio_inputs)
    x = layers.LSTM(lstm_units,return_sequences=True, name=mod+'_lstm1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(lstm_units//2, name=mod+'_lstm2')(x)
    x = layers.Dropout(0.2)(x)

    #create first set of V/A outputs with one lstm layer
    output = layers.Dense(1, activation='tanh', name=mod+'_arousal-valence')(x)

    model = models.Model(audio_inputs, outputs=output, name=mod+"_lstm")
    
    
    return model

def modelBuilder_LSTM_sep(seq_len, dense_units, lstm_units, numOfFes_a, numOfFes_v):
    """
    define LSTM model architecture using Keras' functional API. This one is for emobase features (988 audio features)
    """

    #LSTM with openSMILE audio features as input
    audio_inputs = layers.Input(shape=(seq_len, numOfFes_a), name="audio_features")
    video_inputs = layers.Input(shape=(seq_len, numOfFes_v), name="muted_features")

    #call a layer on model_inputs, in this case applying Dense layer to each timestep 
    xa = layers.TimeDistributed(layers.Dense(dense_units))(audio_inputs)
    xa = layers.LSTM(lstm_units,return_sequences=True)(xa)
    xa = layers.Dropout(0.2)(xa)
    xa = layers.LSTM(lstm_units//2)(xa)
    xa = layers.Dropout(0.2)(xa)
    
    xv = layers.TimeDistributed(layers.Dense(dense_units))(video_inputs)
    xv = layers.LSTM(lstm_units,return_sequences=True)(xv)
    xv = layers.Dropout(0.2)(xv)
    xv = layers.LSTM(lstm_units//2)(xv)
    xv = layers.Dropout(0.2)(xv)
    
    #concat
    x = layers.concatenate([xa,xv])
    #dense layer
    x = layers.Dense(dense_units,activation='relu')(x)

    #create first set of V/A outputs with one lstm layer
    output = layers.Dense(1, activation='tanh', name='out_var1')(x)

    model = models.Model(inputs = [audio_inputs, video_inputs], outputs=output, name="lstm")
    
    return model


def modelLoader_LSTM_all(audio_model, muted_model, dense_units):
    
    audio_inputs = audio_model.get_layer('audio_features').input
    
    timedense_a = audio_model.get_layer('audio_dense')
    xa = timedense_a(audio_inputs)
    
    lstm1a = audio_model.get_layer('audio_lstm1')
    xa = lstm1a(xa)
    xa = layers.Dropout(0.2)(xa)
    lstm2a = audio_model.get_layer('audio_lstm2')
    xa = lstm2a(xa)
    xa = layers.Dropout(0.2)(xa)
    
    video_inputs = muted_model.get_layer('muted_features').input

    timedense_v = muted_model.get_layer('muted_dense')
    xv = timedense_v(video_inputs)    
    
    lstm1v = muted_model.get_layer('muted_lstm1')
    xv = lstm1v(xv)
    xv = layers.Dropout(0.2)(xv)
    lstm2v = muted_model.get_layer('muted_lstm2')
    xv = lstm2v(xv)
    xv = layers.Dropout(0.2)(xv)
    
    #concat
    x = layers.concatenate([xa,xv])
    #dense layer
    x = layers.Dense(dense_units,activation='relu')(x)

    #create first set of V/A outputs with one lstm layer
    output = layers.Dense(1, activation='tanh', name='out_var1')(x)

    model = models.Model(inputs = [audio_inputs, video_inputs], outputs=output)
    
    return model


def load_opensmile(video_id, directory_audio):
    filename = directory_audio + '/' + video_id + '.csv'
    data = pd.read_csv(filename)

    return data


def keras_load_data_music(av_df, urls, directory, seq_len, ewe):
    """
    MUSIC ONLY
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_, y_a, y_v = [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        data = load_opensmile(url, directory)

        #length=118 for training
        data = data.truncate(after=117)
        data = data.iloc[:, 2:(len(data.columns)-1)] #drop columns from opensmile output not required for model training

        for j in range(len(data)-seq_len):
            data_.append(np.array(data[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='music'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='music'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_ = np.array(data_)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_, y_a, y_v


def keras_load_music_video(av_df, urls, directory, seq_len, ewe):
    """
    MUSIC ONLY
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_, y_a, y_v = [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        data = load_opensmile(url, directory)

        #length=118 for training
        data = data.truncate(after=117)
        data = data.iloc[:, 2:(len(data.columns)-1)] #drop columns from opensmile output not required for model training

        for j in range(len(data)-seq_len):
            data_.append(np.array(data[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_ = np.array(data_)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_, y_a, y_v


def get_kfold_cross_validation(splits_nr=5):
    """
    define common split function for baseline model cross-validation
    """
    kfold = sklearn.model_selection.KFold(n_splits=splits_nr, shuffle=True, random_state=42)

    return kfold


def getEWE(data):
    """
    calculate Evaluator Weighted Estimator as detailed in prof Desmond's paper. 
    """

    ratingsArray_arousal, ratingsArray_valence  = [], []

    for n, g in data.groupby('participant_id'):
        ratingsArray_arousal.append(g.iloc[0:118].arousal)
        ratingsArray_valence.append(g.iloc[0:118].valence)

    averagedRating_arousal = np.array(ratingsArray_arousal).mean(axis=0)
    std_arousal = np.array(ratingsArray_arousal).std(axis=0)

    averagedRating_valence = np.array(ratingsArray_valence).mean(axis=0)
    std_valence = np.array(ratingsArray_valence).std(axis=0)

    #weight each individual's ratings
    weightedRatings_arousal, weightedRatings_valence = [], []
    weights_arousal, weights_valence = [], []

    for n, g in data.groupby('participant_id'):   
        weight_arousal, _ = stats.pearsonr(g.iloc[0:118].arousal, averagedRating_arousal)
        
        #there's a slightly worrisome no. of ratings with corr < 0 (arousal: 136, valence: 172), but I don't see what can be done...
        if weight_arousal > 0:
            weightedRatings_arousal.append(g.iloc[0:118].arousal * weight_arousal)
            weights_arousal.append(weight_arousal)

        weight_valence, _ = stats.pearsonr(g.iloc[0:118].valence, averagedRating_valence)

        #if not np.isnan(weight_valence):
        if weight_valence > 0:
            weightedRatings_valence.append(g.iloc[0:118].valence * weight_valence)
            weights_valence.append(weight_valence)
    
    ewe_arousal = np.array(weightedRatings_arousal).sum(axis=0) * (1 / sum(weights_arousal))
    ewe_valence = np.array(weightedRatings_valence).sum(axis=0) * (1 / sum(weights_valence))

    return ewe_arousal, ewe_valence, std_arousal, std_valence


def getCCC(rating1, rating2):
    """
    calculate ccc between two ratings sequences. 
    """
    rating1 =  np.array(rating1)
    rating2 =  np.array(rating2)

    ccc = concordance_cc(torch.tensor(rating1).reshape(-1), torch.tensor(rating2).reshape(-1))
    return np.array(ccc)[0]



def getMeanRatings(data):
    """
    calculate naively averaged ratings.
    """  
    meanData = data.groupby('seq').mean()
    
    return meanData.arousal.iloc[0:118], meanData.valence.iloc[0:118]


def scale_lstm_features(X_train, X_val, X_test):
    """
    Scaler for 3D LSTM arrays.
    """
    #define n_features to reshape data into 2d data for MinMaxScaler
    seq_len, train_features = X_train.shape[1], X_train.shape[2]

    #scale features
    X_train = np.reshape(X_train, (-1, train_features))
    X_val = np.reshape(X_val, (-1, train_features))
    X_test = np.reshape(X_test, (-1, train_features))

    X_train, X_val, X_test = scale_lm_features(X_train, X_val, X_test)

    #reshape into 3d arrays
    X_train = np.reshape(X_train, (-1, seq_len, train_features))
    X_val = np.reshape(X_val, (-1, seq_len, train_features))
    X_test = np.reshape(X_test, (-1, seq_len, train_features))

    return X_train, X_val, X_test


def lstm_scoring(model, X_test, y_test):
    """
    calculates MSE, Pearson's r and concordance correlation coefficient of an LSTM's output predictions
    """
    preds = model.predict(X_test)

    mse = model.evaluate(X_test, y_test)
    ccc = concordance_cc(torch.tensor(y_test).reshape(-1), torch.tensor(preds).reshape(-1))

    return mse[0], np.array(ccc)[0]

def lstm_scoring_twoInps(model, X_test_a, X_test_v, y_test):
    """
    calculates MSE, Pearson's r and concordance correlation coefficient of an LSTM's output predictions
    """
    preds = model.predict([X_test_a,X_test_v])

    mse = model.evaluate([X_test_a,X_test_v], y_test)
    ccc = concordance_cc(torch.tensor(y_test).reshape(-1), torch.tensor(preds).reshape(-1))

    return mse[0], np.array(ccc)[0], preds


def scale_lm_features(X_train, X_val, X_test):
    """
    Fit scaler on training data and return scaled train and validation data.
    """
    scaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test

def keras_load_data_video(av_df, urls, directory_video, seq_len, ewe):
    """
    MUTED VIDEO
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_, y_a, y_v = [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        data = load_videofeatures(url, directory_video)

        #length=118 for training
        data.drop(data.tail(2).index,inplace=True)
        data = data.iloc[:, 2:(len(data.columns)-1)] #drop columns from opensmile output not required for model training

        for j in range(len(data)-seq_len):
            data_.append(np.array(data[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='muted_video'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='muted_video'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_ = np.array(data_)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_, y_a, y_v


def keras_load_muted_video(av_df, urls, directory_video, seq_len, ewe):
    """
    MUTED VIDEO
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_, y_a, y_v = [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        data = load_videofeatures(url, directory_video)

        #length=118 for training
        data.drop(data.tail(2).index,inplace=True)
        data = data.iloc[:, 2:(len(data.columns)-1)] #drop columns from opensmile output not required for model training

        for j in range(len(data)-seq_len):
            data_.append(np.array(data[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_ = np.array(data_)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_, y_a, y_v


def keras_load_data_all(av_df, urls, directory_video, seq_len, ewe, directory):
    """
    AUDIO AND VIDEO
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_, y_a, y_v = [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        #for video first
        datav = load_videofeatures(url, directory_video)

        #length=118 for training
        datav.drop(datav.tail(2).index,inplace=True)
        datav = datav.iloc[:, 2:(len(datav.columns)-1)].to_numpy() #drop columns from opensmile output not required for model training
        #add audio
        dataa = load_opensmile(url, directory)

        #length=118 for training
        dataa = dataa.truncate(after=117)
        dataa = dataa.iloc[:, 2:(len(dataa.columns)-1)].to_numpy() #drop columns from opensmile output not required for model training
        #concat
        data = np.concatenate([datav,dataa],axis = 1)
        data = pd.DataFrame(data)
        for j in range(len(data)-seq_len):
            data_.append(np.array(data[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(data)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(data)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_ = np.array(data_)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_, y_a, y_v

def keras_load_data_all_sep(av_df, urls, directory_video, seq_len, ewe, directory):
    """
    AUDIO AND VIDEO
    load 3d arrays for input to LSTM. Emotion==2 for arousal, emotion==3 for valence (I know this code is terrible)
    """
    data_a, data_v, y_a, y_v = [], [], [], []

    #requires that working directory be one level above the repo's
    for url in urls.video_id:
        media_id = url.split('_')[0]
        #for video first
        datav = load_videofeatures(url, directory_video)

        #length=118 for training
        datav.drop(datav.tail(2).index,inplace=True)
        datav = datav.iloc[:, 2:(len(datav.columns)-1)].to_numpy() #drop columns from opensmile output not required for model training
        #add audio
        dataa = load_opensmile(url, directory)

        #length=118 for training
        dataa = dataa.truncate(after=117)
        dataa = dataa.iloc[:, 2:(len(dataa.columns)-1)].to_numpy() #drop columns from opensmile output not required for model training
        
        datav = pd.DataFrame(datav)
        dataa = pd.DataFrame(dataa)
        for j in range(len(datav)-seq_len):
            data_a.append(np.array(dataa[j:(j + seq_len)]))
            data_v.append(np.array(datav[j:(j + seq_len)]))

        #keep last arousal-valence value of each window as the prediction target
        y_arousal = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][2])
        y_valence = pd.Series([x for x in ewe if x[0]==media_id if x[1]=='video'][0][3])

        y_arousal = y_arousal.truncate(before=len(y_arousal)-118+seq_len).reset_index(drop=True)
        y_arousal = y_arousal.truncate(after=len(dataa)-seq_len-1)

        y_valence = y_valence.truncate(before=len(y_valence)-118+seq_len).reset_index(drop=True)
        y_valence = y_valence.truncate(after=len(dataa)-seq_len-1)

        y_a.extend(y_arousal)
        y_v.extend(y_valence)
        #print(len(y_arousal))

    data_a = np.array(data_a)
    data_v = np.array(data_v)
    y_a = np.array(pd.DataFrame(y_a))
    y_v = np.array(pd.DataFrame(y_v))

    return data_a, data_v, y_a, y_v


def load_videofeatures(video_id, directory_video):

    datav = pd.read_csv(directory_video)
    datav = datav.loc[datav['media_id'] == video_id]
    return datav
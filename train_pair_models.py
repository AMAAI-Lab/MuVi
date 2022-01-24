#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sklearn
import numpy as np
import pandas as pd

import tensorflow as tf
#uncomment if there is no GPU available
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import os.path
from scipy import stats
from aux_files import getEWE, train_lstm, get_kfold_cross_validation, print_results

'''TRAIN ALL LSTM MODELS'''

'''1. Pre Processing the DATASET'''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#path
data_folder = "./"

#retrieve urls, participant, media and av data
#video urls contains the youtube ids of MuVi dataset
urls = pd.read_csv(os.path.join(data_folder, "video_urls.csv"))
#participant data contains information regarding the background (demographics) of the users
participant_df = pd.read_csv(os.path.join(data_folder, "participant_data.csv"))
#media data contains the static annotations which describe the media item’s overall emotion
media_df = pd.read_csv(os.path.join(data_folder, "media_data.csv"))
#av_data includes the dynamic (continuous) annotations for Valence and Arousal.
av_df = pd.read_csv(os.path.join(data_folder, "av_data.csv"))


#drop group in av_df with length==111
drop_indexes = av_df[(av_df.participant_id == 23) & (av_df.media_id == 'U0CGsw6h60k')].index
av_df.drop(drop_indexes, inplace=True)

#create rowIDs for av_df
av_df['rowID'] = list(zip(av_df["media_id"], av_df["media_modality"], av_df["participant_id"]))
media_df['rowID'] = list(zip(media_df["media_id"], media_df["media_modality"], media_df["participant_id"]))

#drop low-quality annotations (cursor not moved at all)
constant_inputs = []
i=0

for name, group in av_df.groupby(['media_id','media_modality','participant_id']):
    p, _ = stats.pearsonr(group.iloc[0:118].arousal, group.iloc[0:118].valence)
    
    if np.isnan(p):
        constant_inputs.append(name)
    
    elif p < 0:
        i+=1

av_df = av_df[~av_df.rowID.isin(constant_inputs)]
media_df = media_df[~media_df.rowID.isin(constant_inputs)]

#retrieve media_df colnames
emotions = np.array(media_df.iloc[:, 4:31].columns)

#merge dfs for later analysis
participant_av_df = pd.merge(av_df, media_df.iloc[:,0:4], on=['participant_id', 'media_id', 'media_modality'])
participant_av_df = pd.merge(participant_av_df, participant_df, on=['participant_id'])

participant_media_df = pd.merge(media_df, participant_df, on=['participant_id'])
participant_media_df = participant_media_df.replace({False: 0, True: 1})

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''2. GET EWE'''

#calculate overall EWE ratings for all media items
ewe = []

for name, group in av_df.groupby(['media_id','media_modality']):
    a, v, a_std, v_std = getEWE(group)
    ewe.append([name[0], name[1], a, v, a_std, v_std])
    
    
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''3. SET PARAMETERS FOR THE PAIR MODELS'''

#set audio and visual features
opensmileFeatureDir = './emobase_features/' 
video_features = 'combined_visual_features.csv'

#hyper-parameters based on the paper
hp = {
    'seq_len': 4, #sequence length
    'dense_units': 256, #units for dense and lstm blocks
    'lstm_units': 256,
    'lr': 0.0001
}

#set cross validation
kf = get_kfold_cross_validation()

'''4. SET MODALITY and ARCHITECTURE '''
#####################
'''Use one of the following setups based on the experimental setup of the paper.
Set the variable "modality" according to which model you want to train. A1, A2 
and PAIR architectures are the last ones'''

#1. 'music' (Music Modality, Audio Features )
#2. 'muted_video' (Video Modality, Video Features)
#3. 'video_music' (AudioVisual Modality, Audio Features)
#4. 'video_muted' (AudioVisual Modality, Audio Features)
#PAIR ARCHITECTURES
#5. 'musicvideoConcat' (AudioVisual Modality, Audio Features, A1 EARLY FUSION)
#6. 'musicvideoSep' (AudioVisual Modality, Audio Features, A2 LATE FUSION)
#7. 'musicvideoShared' (AudioVisual Modality, Audio Features, PAIR)
#####################
modality = 'music'    


#create splits
allsplits = []
for train_index, valtest_index in kf.split(urls):
    #split the data to train, validation, testing
    val_index, test_index = sklearn.model_selection.train_test_split(valtest_index, random_state=42, test_size=0.5)
    allsplits.append([train_index,val_index,test_index])
#Call the training process for all splits. 
#Returns a dataframe regarding the number of split, emotion and mse and ccc predictions
results = train_lstm(hp, opensmileFeatureDir, video_features, ewe, modality, urls, av_df, allsplits)

'''5. PRINT RESULTS '''  
print_results(results)
 



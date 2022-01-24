# MuVi
Predicting emotion from music videos: exploring the relative contribution of visual and auditory information on affective responses.

## Dataset files
-video_urls.csv: Contains the YouTube ids of MuVi dataset. We can also provide all the media files (for all modalities) upon e-mail request. <br />
-participant_data.csv: We provide the anonymized profile and demographic information of the annotators. <br />
-media_data.csv:  Contains the static annotations which describe the media item’s overall emotion. The terms that were used are based on the GEMS-28 term list. <br /> 
-av_data.csv: Includes the dynamic (continuous) annotations for Valence and Arousal. <br />
<br />
We also include the extracted audio (./emobase_features folder) features. Visual features can be found [here.](https://drive.google.com/file/d/1avyXoSi1mXPONwInKu0hBWbXCHF8CgjC/view?usp=sharing)  <br />

## Train PAIR models
train_pair_models.py includes all the functions and information for pre-processing and training the proposed PAIR architectures. <br />

#### Prerequisites
sklearn <br />
numpy <br />
pandas <br />
tensorflow <br />
torch <br />
scipy <br />
audtorch <br />

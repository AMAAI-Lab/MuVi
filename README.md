# Music Videos (MuVi)
This is the dataset and models to accompany the [paper](https://arxiv.org/abs/2202.10453) by Phoebe Chua, Dr. Dimos Makris, Prof. Dorien Herremans, Prof. Gemma Roig, and Prof. Kat Agres on Predicting emotion from music videos: exploring the relative contribution of visual and auditory information on affective responses.

**Abstract:**
_Although media content is increasingly produced, distributed, and consumed in multiple combinations of modalities, how individual modalities contribute to the perceived emotion of a media item remains poorly understood. In this paper we present MusicVideos (MuVi), a novel dataset for affective multimedia content analysis to study how the auditory and visual modalities contribute to the perceived emotion of media. The data were collected by presenting music videos to participants in three conditions: music, visual, and audiovisual. Participants annotated the music videos for valence and arousal over time, as well as the overall emotion conveyed. We present detailed descriptive statistics for key measures in the dataset and the results of feature importance analyses for each condition. Finally, we propose a novel transfer learning architecture to train Predictive models Augmented with Isolated modality Ratings (PAIR) and demonstrate the potential of isolated modality ratings for enhancing multimodal emotion recognition. Our results suggest that perceptions of arousal are influenced primarily by auditory information, while perceptions of valence are more subjective and can be influenced by both visual and auditory information. The dataset is made publicly available._

If you find this resource useful, [please cite the original work](https://arxiv.org/abs/2202.10453):

      @article{chua2022predicting,
        title={Predicting emotion from music videos: exploring the relative contribution of visual and auditory information to affective responses},
        author={Chua, Phoebe and Makris, Dimos and Herremans, Dorien and Roig, Gemma and Agres, Kat},
        journal={arXiv preprint arXiv:2202.10453},
        year={2022}
      }

  Chua, P., Makris, D., Herremans, D., Roig, G., & Agres, K. (2022). Predicting emotion from music videos: exploring the relative contribution of visual and auditory information to affective responses. arXiv preprint arXiv:2202.10453.


## Dataset files
- `video_urls.csv`: contains the YouTube ids of MuVi dataset. We can also provide all the [media files](https://zenodo.org/record/7127775#.Y0Dq1exBxhE) for all modalities.
- `participant_data.csv`: we provide the anonymized profile and demographic information of the annotators.
- `media_data.csv`:  contains the static annotations which describe the media item’s overall emotion. The terms that were used are based on the GEMS-28 term list.
- `av_data.csv`: \iIncludes the dynamic (continuous) annotations for valence and arousal.


We also included the extracted audio features (``./emobase_features``). The extracted audio features as well as all extracted visual features (color, lighting key, facial expressions, scenes, objects and actions) can be found [here.](https://zenodo.org/record/7128177)

## PAIR models
The file `train_pair_models.py` includes all of the functions and information necessary for pre-processing and training the proposed PAIR architectures.

## Prerequisites
- sklearn
- numpy
- pandas
- tensorflow
- torch
- scipy
- audtorch

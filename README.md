
The EQN framework is a micro-emotion annotation and detection system that realizes the automatic micro-emotion annotation of text with energy level scores for the first time. The text emotion datasets it annotates are no longer simple single-label or multi-label, but macro-emotions and micro-emotions with continuous values ​​of emotion intensity. The labeling of emotion datasets has changed from discrete to continuous. It plays an important role in the subtle research of emotions in fields such as emotional computing, human-computer alignment, humanoid robots, and psychology.
This is the experimental result of the EQN micro-emotion detection and annotation framework we proposed, the train.csv of the Goemotions dataset with micro-emotion labels with energy level intensity values
and the model trained on the Goemotions dataset based on the BERT model. Attached is the micro-emotion annotation code based on pytorch, which can be used to annotate the Goemotions dataset by yourself, or predict the emotion classification based on the annotation results. For the specific implementation method, please refer to our paper


Note:
1. gotrainadd.csv: Goemotions dataset with additional annotation (micro-emotion labels with energy level intensity values(0-10)).
2. 28pd.py: Micro-emotion detection and annotation code based on pytorch.
3. 55770-1.pth: Model trained on the Goemotions dataset based on the BERT model (emotion energy level intensity is a value between 0-1).
4. Goemotions dataset: Data and code available at https://github.com/google-research/google-research/tree/master/goemotions

The experimental environment of this project.
GPU：NVIDIA GeForce RTX 3090 GPU
Bert-base-cased pre-trained model: https://huggingface.co/google-bert/bert-base-cased
python=3.7，pytorch=1.9.0，cudatoolkit=11.3.1，cudnn=8.9.7.29.

Instructions for use:

1. Refer to our usage environment instructions and install the operating environment.
2. Download our EQN-model.
3. Change the loading model name in 28pd.py to the actual name of the downloaded EQN-model.
4. Create a directory named "28pd" to place the .csv format data files to be labeled or predicted.


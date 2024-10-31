# Expansion-Quantization-Network-EQN
EQN框架是一个微情感标注和检测系统，首次实现了带能级分数的文本自动微情感标注。其标注的文本情感数据集，不再是简单的单标签、多标签，而是宏情感、微情感带有情感强度连续数值的。情感数据集的标记从离散化进入连续化。对情感计算、人机对齐、人形机器人和心理学等领域对情感的细微研究具有重要作用。

这是我们提出的EQN微情感检测和标注框架的实验结果、补充标注Goemotions数据集的train.csv带有能级强度数值的微情感标签。
和基于BERT模型在Goemotions数据集上训练的模型。附有基于pytorch的微情感标注代码，可自行对Goemotions数据集进行标注，或根据标注的结果预测情感分类。具体的分类方法稍后参见我们即将发布的论文。

说明·：
1. gotrainadd.csv ：补充标注的Goemotions数据集（带有能级强度数值的微情感标签）。
2. 28pd.py ：基于pytorch的微情感检测和标注代码。
3. 55770-1.pth：基于BERT模型在Goemotions数据集上训练的模型(情感能级强度在0-1之间的数值)。
4. Goemotions dataset: Data and code available at https://github.com/google-research/google-research/tree/master/goemotions


本项目实验环境。
GPU：NVIDIA GeForce RTX 3090 GPU
Bert-base-cased预训练模型: https://huggingface.co/google-bert/bert-base-cased
python=3.7，pytorch=1.9.0，cudatoolkit=11.3.1，cudnn=8.9.7.29.

The EQN framework is a micro-emotion annotation and detection system that realizes the automatic micro-emotion annotation of text with energy level scores for the first time. The text emotion datasets it annotates are no longer simple single-label or multi-label, but macro-emotions and micro-emotions with continuous values ​​of emotion intensity. The labeling of emotion datasets has changed from discrete to continuous. It plays an important role in the subtle research of emotions in fields such as emotional computing, human-computer alignment, humanoid robots, and psychology.
This is the experimental result of the EQN micro-emotion detection and annotation framework we proposed, the train.csv of the Goemotions dataset with micro-emotion labels with energy level intensity values
and the model trained on the Goemotions dataset based on the BERT model. Attached is the micro-emotion annotation code based on pytorch, which can be used to annotate the Goemotions dataset by yourself, or predict the emotion classification based on the annotation results. The specific classification method will be referred to in our upcoming paper later.

Note:
1. gotrainadd.csv: Goemotions dataset with additional annotation (micro-emotion labels with energy level intensity values).
2. 28pd.py: Micro-emotion detection and annotation code based on pytorch.
3. 55770-1.pth: Model trained on the Goemotions dataset based on the BERT model (emotion energy level intensity is a value between 0-1).
4. Goemotions dataset: Data and code available at https://github.com/google-research/google-research/tree/master/goemotions

The experimental environment of this project.
GPU：NVIDIA GeForce RTX 3090 GPU
Bert-base-cased pre-trained model: https://huggingface.co/google-bert/bert-base-cased
python=3.7，pytorch=1.9.0，cudatoolkit=11.3.1，cudnn=8.9.7.29.

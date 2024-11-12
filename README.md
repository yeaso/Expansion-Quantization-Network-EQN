# Expansion-Quantization-Network-EQN
EQN框架是一个微情感标注和检测系统，首次实现了带能级分数的文本自动微情感标注。其标注的文本情感数据集，不再是简单的单标签、多标签，而是宏情感、微情感带有情感强度连续数值的。情感数据集的标记从离散化进入连续化。对情感计算、人机对齐、人形机器人和心理学等领域对情感的细微研究具有重要作用。
![003](https://github.com/user-attachments/assets/f99bae3e-fb1a-49ac-b45d-e4080016869a)


这是我们提出的EQN微情感检测和标注框架的实验结果、补充标注Goemotions数据集的train.csv带有能级强度数值(0-10)的微情感标签。
和基于BERT模型在Goemotions数据集上训练的模型。附有基于pytorch的微情感标注代码，可自行对Goemotions数据集进行标注，或根据标注的结果预测情感分类。具体的分类方法参见我们的论文
【Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework】https://arxiv.org/abs/2411.06160

说明·：
1. gotrainadd.csv ：补充标注的Goemotions数据集（带有能级强度数值(0-10)的微情感标签）。
2. 28pd.py ：基于pytorch的微情感检测和标注代码。
3. 55770-1.pth：基于BERT模型在Goemotions数据集上训练的模型(情感能级强度在0-1之间的数值)。
4. Goemotions dataset: Data and code available at https://github.com/google-research/google-research/tree/master/goemotions


本项目实验环境。
GPU：NVIDIA GeForce RTX 3090 GPU
Bert-base-cased预训练模型: https://huggingface.co/google-bert/bert-base-cased
python=3.7，pytorch=1.9.0，cudatoolkit=11.3.1，cudnn=8.9.7.29.
EQN基于BERT模型对Goemotions测试集标记情感能级分数，根据情感能级分数计算皮尔逊相关系数后的热力图。
![Figure_1 (1)](https://github.com/user-attachments/assets/a7a646db-471a-4f29-8dd3-ee1080153c90)
EQN annotates the Goemotions test set with sentiment energy level scores based on the BERT model, and calculates the heat map after the Pearson correlation coefficient is calculated based on the sentiment energy level scores.

使用说明：

1. 参考我们的使用环境说明，安装好运行环境。
2. 下载我们的EQN-model。
3. 把28pd.py中的装载模型名称修改为下载的EQN-model的实际名称。
4. 建立一个名称为“28pd”的目录，该目录放置要标注或预测的.csv格式的数据文件。
The EQN framework is a micro-emotion annotation and detection system that realizes the automatic micro-emotion annotation of text with energy level scores for the first time. The text emotion datasets it annotates are no longer simple single-label or multi-label, but macro-emotions and micro-emotions with continuous values ​​of emotion intensity. The labeling of emotion datasets has changed from discrete to continuous. It plays an important role in the subtle research of emotions in fields such as emotional computing, human-computer alignment, humanoid robots, and psychology.
This is the experimental result of the EQN micro-emotion detection and annotation framework we proposed, the train.csv of the Goemotions dataset with micro-emotion labels with energy level intensity values
and the model trained on the Goemotions dataset based on the BERT model. Attached is the micro-emotion annotation code based on pytorch, which can be used to annotate the Goemotions dataset by yourself, or predict the emotion classification based on the annotation results. For the specific implementation method, please refer to our paper
【Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework】 https://arxiv.org/abs/2411.06160

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

Welcome to cite our paper!
[Jingyi Zhou, Senlin Luo, Haofan Chen. "Expansion Quantization Network: An Efficient Micro-emotion Annotation and Detection Framework." arXiv preprint arXiv.2411.06160 (2024).]

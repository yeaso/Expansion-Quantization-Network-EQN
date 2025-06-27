'''
Training Process (Divided into Two Stages):
Initialize Training Data:
Map the original single-label emotion annotations (e.g., "joy") into numerical vectors representing 6 emotion categories (using multi-label one-hot encoding — the annotated category is set to 1, others to 0).

Train Model A and Re-label the Dataset:
Use Model A to make predictions on the training set, applying soft labels (i.e., probability scores) to emotion categories other than the original label.

Train Model B:
Retrain the model using the updated dataset with soft labels to obtain a more robust Model B.

🔧 File Descriptions (for 28-class GoEmotions Dataset):
train.csv: The original training data. Columns are text and labels (e.g., "i feel joy" and "joy").

model_A.pth: Path to save the initial Model A.

model_B.pth: Path to save the refined Model B.


'''



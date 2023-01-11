# Galaxy Finetuning Using Personalized Code

In this folder you will find code regarding CounTR's finetuning starting from the pretrained weights on FSC147 and CARPK.  
The only original code used is the one for the creation of the model.

The content of the folder is the following:

- ./generate_annotations.ipynb 
containing the code to generate the dataset and the train/val/test split in the needed format
- ./training.ipynb
containing the notebook to implement data augmentation and the lightning module to train the model
- ./testing.ipynb  
containing the notebook to test the model on the test split of the best checkpoint and plot the predictions
- ./keypoints_viz.ipynb
containing the notebook to plot a sample image with its keypoints overimpressed
- models_crossvit.py, models_mae_cross.py and util folder
this is the only code used from teh original paper in order to create and load the model from a checkpoint

# Galaxy Finetuning Using CounTR original code

In this folder you will find code regarding CounTR's finetuning starting from FSC147 and CARPK finetuned weights.  
For this purpose most of the code used is paper's original and the file that we created are:

- GALAXY_finetuning.ipynb  
containing simply logs of finetuning and testing of the model 
- CounTR/GALAXY_finetune.py
containing the script called for finetuning. note that most of this code is the same as their FSC_finetune_cross.py
- CounTR/GALAXY_testing.py  
containing the script called for testing. Once again, most of the code was made by the paper's authors. 
- CounTR/util/galaxy_augmentation.py
containing the script to apply their data augmentation pipeline to our data. In the report we mentioned how me modified it to test effectiveness of their code to our task.

- models_mae_cross.py, models_crossvit.py and the util folder to build the architecture 

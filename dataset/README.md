## Download datasets


To download datasets `cd` to this folder and use ```bash dowlaoad_datasets.sh```.\
You will get:
1) backgrounds.zip for making synthetic data for cleaning module  
2) abc.zip datasets for abc containing 3 folders train,test,validation. Each folder 
 contains .svg files with wich are used to train our model reported in paper.
3) Dataset_of_cleaning.zip  on which we fine-tuned our cleaning model
4) precision_floorplan folder with pdf files in it
5) golden_set.zip in wich you could find files on wich we evaulated in our paper.
Please be aware that files from golden set is subset from  Dataset_of_cleaning.zip . 
You should exclude those data from training or validation. This files only for testing. 

For abc and precision floorplans dataset preprocessing please look at dataset_utils/ folder (read ReadMe there). 
For datasets of cleaning please take a look at cleaning utils for generating data and dataloader.

You can chose which file to download by commenting code in download_datasets.sh

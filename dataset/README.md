## Download datasets


To download datasets `cd` to this folder and use ```bash dowlaoad_datasets.sh```.\
You will get:
1) backgrounds.zip for making synthetic data for cleaning module  
2) abc.zip datasets for abc containing 3 folders train,test,validation. Each folder 
 contain .svg files with wich are used to train our model reported in paper.
3) Dataset of cleaning on wich we finetuned our cleaning model
4) precision_floorplan folder with pdf files in it
 

For abc and precision floorplans dataset preprocessing please look at utils/data folder.
For datasets of cleaning please take a look at cleaning utils for generating data and dataloader.

You can chose wich file to download by commenting code in download_datasets.sh

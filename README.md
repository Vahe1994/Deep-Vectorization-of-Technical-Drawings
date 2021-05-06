## Deep Vectorization of Technical Drawings [[Web page](http://adase.group/3ddl/projects/vectorization/)] [[Paper](https://arxiv.org/abs/2003.05471)] [[Video](https://www.youtube.com/watch?v=lnQNzHJOLvE)] [[Slides](https://drive.google.com/file/d/1ZrykQeA2PE4_8yf1JwuEBk9sS4OP8KeM/view?usp=sharing)]
Official Pytorch repository for ECCV 2020 paper [Deep Vectorization of Technical Drawings](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_35)

![alt text](https://drive.google.com/uc?export=view&id=191r0QAaNhOUIaHPOlPWH5H4Jg7qxCMRA) 
## IMPORTANT
 
The project is constantly updating to add more future and make a code easier to understand. 
In the table below, you can see the current state of the repository and future updates.

| Modules       | Added         | Refactored| Tested|
| ------------- |:-------------:| :-----:| -----:|
| vectorization | partly        |    partly |    No |
| loss functions| yes           |   yes |   yes |
| cleaning      | yes           |    yes |    No |
| refinement    | yes           |    partly |    No |
| data_scripts  | yes           |    yes |    No |
| datasets      | yes           |   yes |    No |
| image degradation| yes        |   No  |    yes |
| merging       | yes           |   No  |    No  |
| notebooks     | yes            |    partly |    yes |
| utils         | partly        |    No |    No |
| rendering     | yes           |   yes |   yes |
| metrics       | partly        |    No |    No |
| trained models| yes            |    No |    No |
| dockers       | No            |    No |    No |
| script to run | No            |    No |    No |
| Readme's | partly            |    No |    No |
| documentation | partly        |    No |    No |
| requirement    | No        |    No |    No |

## Repository Structure

To make the repository user-friendly, we decided to stick with - module-like structure.
The main modules are cleaning, vectorization, refinement, and merging(each module has an according to folder).
Each folder has Readme with more details. Here is the brief content of each folder.

* cleaning - model, script to train and run, script to generate synthetic data 
* vectorization - NN models, script to train
* refinement - refinement module for curves and lines
* merging - merging module for curves and lines
* dataset - scripts to download ABC, PFP, cleaning datasets, scripts to modify data into patches, and memory-mapped them.
* notebooks - a playground to show some function in action
* utils - loss functions, rendering, metrics
* scripts - scripts to run training and evaluation

## Requirments
Linux system \
Python 3
Pytorch 1.3 + \
tochvision \
rtree \
conda \
cairocffi


## Compare 

To compare with us without running code, you can download our results on the full pipeline on the test set
for [pfp](https://drive.google.com/file/d/1FGm-JQsvOa5sbi_f_-MMl1XC5Z8JGe0F/view?usp=sharing) and for 
[abc](https://drive.google.com/file/d/1lR5lea3sY4Bhp9QL4MmmPs0kqZ5voPGu/view?usp=sharing).


## Dataset
Scripts to download dataset are in folder dataset/.
* For ABC,real datasets use download_dataset.sh
* For PFP, use precision_floorplan_download.py  
Read ReadMe there for more instructions.

## Notebooks 

To show how some of the usability of the functions, there are several notebooks in the notebooks folder.
1) Rendering notebook
2) Dataset loading, model loading, model training, loss function loading
3) Notebook that illustrates  how to work with pretrained model and how to do refinement on lines(without merging) 
4) Notebook that illustrates how to work with pretrained model and how to do refinement on curves(without merging)

## Models

Download pretrained models for [curve](https://drive.google.com/file/d/18jN37pMvEg9S05sLdAznQC5UZDsLz-za/view?usp=sharing)
and for [line](https://drive.google.com/file/d/1Zf085V3783zbrLuTXZxizc7utszI9BZR/view?usp=sharing) .

## How to run 
Look at notebooks pretrain_model_loading_and_evaluation_for_line.ipynb and 
pretrain_model_loading_and_evaluation_for_curve.ipynb , for an example how to run primitive estimation 
and refinement for curve and line.  



## How to train
Look at vectorization/srcipts/train_vectorizatrion (currently under refactoring)

### Citing
```
@InProceedings{egiazarian2020deep,
  title="Deep Vectorization of Technical Drawings",
  author="Egiazarian, Vage and Voynov, Oleg and Artemov, Alexey and Volkhonskiy, Denis and Safin, Aleksandr and Taktasheva, Maria and Zorin, Denis and Burnaev, Evgeny",
  booktitle="Computer Vision -- ECCV 2020",
  year="2020",
  publisher="Springer International Publishing",
  address="Cham",
  pages="582--598",
  isbn="978-3-030-58601-0"
}
```

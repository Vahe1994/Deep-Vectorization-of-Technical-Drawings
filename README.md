## Deep-Vectorization-of-Technical-Drawings | [Webpage](http://adase.group/3ddl/projects/vectorization/) | [Paper](https://arxiv.org/abs/2003.05471) | [Video](https://www.youtube.com/watch?v=lnQNzHJOLvE&t=15s)
Official Pytorch repository for ECCV 2020 [Deep Vectorization of Technical Drawings]()

## IMPORTANT
Because, this project is massive and I'm trying to do user friendly repository.
Not all functions are yet added and tested.Sorry for  inconvenience.
I'm currently working on it.See table below.\

| Modules       | Added         | Cool  |
| ------------- |:-------------:| -----:|
| vectorization | partly        |    No |
| refinement    | partly        |    No |
| data_scripts  | No            |    No |
| datasets      | Yes           |    No |
| merging       | No            |    No |
| notebooks     | No            |    No |
| utils         | partly        |    No |
| dockers       | No            |    No |
| Readme        | No            |    No |

## Repository Structure

To increase user friendly for changes we decided to make repository module like.
The main modules are cleaning,vectorization,refinement and merging(each module has according folder).
Each folder has readme with more details. Here is brief content of each folder.

* cleaning - model,script to train and run, script to generate synthetic data 
* vectorization - Transformer model, script to train
* refinement - 
* merging - 
* dataset - 
* notebooks -  
* utils - loss functions, rendering, metrics
* scripts -

## Requirments
Linux system 
Pytorch


## Compare 

If you want to compare with us without running code you can download our results on full pipeline on test set
for [pfp](https://drive.google.com/file/d/1FGm-JQsvOa5sbi_f_-MMl1XC5Z8JGe0F/view?usp=sharing) and for 
[abc](https://drive.google.com/file/d/1lR5lea3sY4Bhp9QL4MmmPs0kqZ5voPGu/view?usp=sharing) .
## How to run 



## How to train 

### BibTeX
```
@article{egiazarian2020deep,
  title={Deep Vectorization of Technical Drawings},
  author={Egiazarian, Vage and Voynov, Oleg and Artemov, Alexey and Volkhonskiy, Denis and Safin, Aleksandr and Taktasheva, Maria and Zorin, Denis and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2003.05471},
  year={2020}
}
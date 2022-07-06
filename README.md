## Deep Vectorization of Technical Drawings [[Web page](http://adase.group/3ddl/projects/vectorization/)] [[Paper](https://arxiv.org/abs/2003.05471)] [[Video](https://www.youtube.com/watch?v=lnQNzHJOLvE)] [[Slides](https://drive.google.com/file/d/1ZrykQeA2PE4_8yf1JwuEBk9sS4OP8KeM/view?usp=sharing)]
Official Pytorch repository for ECCV 2020 paper [Deep Vectorization of Technical Drawings](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_35)

![alt text](https://drive.google.com/uc?export=view&id=191r0QAaNhOUIaHPOlPWH5H4Jg7qxCMRA)

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
Linux system  
Python 3

See requirments.txt and additional packages

cairo==1.14.12  
pycairo==1.19.1  
chamferdist==1.0.0


## Compare 

To compare with us without running code, you can download our results on the full pipeline on the test set
for [pfp](https://drive.google.com/file/d/1FGm-JQsvOa5sbi_f_-MMl1XC5Z8JGe0F/view?usp=sharing) and for 
[abc](https://drive.google.com/file/d/1lR5lea3sY4Bhp9QL4MmmPs0kqZ5voPGu/view?usp=sharing).


## Dataset
Scripts to download dataset are in folder dataset/.
* For ABC ,real datasets download [here](https://drive.google.com/file/d/1hET43eM2cfwfqI7g1VsusU1wNpvc2VHl/view?usp=sharing) or use scriptdownload_dataset.sh
* For PFP, use precision_floorplan_download.py  
Read ReadMe there for more instructions.
* Real dataset for cleaning download [here](https://drive.google.com/file/d/1dgJLgtPvk9SK9rOCnw-WQRBG6OsMUEH3/view?usp=sharing) or use script download_dataset.sh
* Synthetic datset  generation script for cleaning can be found in cleaning/scripts.
* 
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
1. Download models.
2. Either use Dockerfile to create docker image with needed environment or just install requirements
3. Run scripts/run_pipeline.sh with correct paths for trained model, data dir and output dir. Don't forget to chose primitive type and primitive count in one patch.

P.s. currently cleaning model not included there.
   
## Dockerfile 

Build the docker image:

```bash
docker build -t Dockerfile owner/name:version .
```
example:
```bash
docker build -t vahe1994/deep_vectorization:latest .
```


When running container mount folder with reporitory into code/, folder with datasets in data/ folder with logs in logs/
```bash
docker run --rm -it --shm-size 128G -p 4045:4045 --mount type=bind,source=/home/code,target=/code --mount type=bind,source=/home/data,target=/data --mount type=bind,source=/home/logs,target=/logs  --name=container_name owner/name:version /bin/bash
```

Anaconda with packages are installed in follder opt/ . Environement with packages that needed are installed in environment vect-env.
. To activate it run in container
```bash
. /opt/.venv/vect-env/bin/activate/
```

## How to train
Look at vectorization /srcipts/train_vectorizatrion 

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

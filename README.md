# TrainModel-Container 

## Introduction

> This container will load images specifically "PIXELDATA" from a dicom and match it to is given classification vector. Then Using a defult architecture it will build and train a model over a given set of gridsearch parameters. There are a multitude of parameters that the user is able to modify for generating the data set componenets of the model. If you choose to load a model you can use other architectures however it is not guaranteed that it will work. 

##  Design: 
  * Used python 
  * full list of packages needed: (listed within the Dockerfile.base)
    * pandas 
    * numpy 
    * matplotlib 
    * opencv-python 
    * python-math 
    * pydicom 
    * tensorflow 
    * scikit-learn 
    * tensorflow-addons 
    * pylibjpeg 
    * pylibjpeg-libjpeg 
    * python-gdcm 
    * tqdm 
    * keras 
    * imbalanced-learn 
   
##  How to use:
  > All the scripts are located within the "workspace" dir - any edits you will need to make for your specific use case will be with "model.py". Once edits are done run ./build.sh to build your docker container. Specifics to edit within docker are the Dockerfile.base file for naming the container, pushing to git and libraries used. If you want integration with XNAT navigate to the "xnat" folder and edit the command.json documentation available at @ https://wiki.xnat.org/container-service/making-your-docker-image-xnat-ready-122978887.html#MakingyourDockerImage%22XNATReady%22-installing-a-command-from-an-xnat-ready-image

## Running (ON XNAT): 
  * Navigate to the project on mirrir and click on "Run containers"  
  * The container should show up as "Runs training and testing of model with project mounted" and click it 
  * Fill out necessary arguments and hit run 
  * Will work as just python script convert to jupyternotebook and run on there. 

## Running in general: 
  * model.py loads data from mortality.csv or what ever named .csv you provided, makes arrays for trining/testing/val and training and then runs training, as well as testing
  * For my use cases I have dockersized it so I could run with access to GPU, your usecases may vary  
  * There are arguments needed to run this pipline which can be found within the model.py script 

## NOTES: 
  * If you do not click save the model will NOT be saved it will just run and results will be printed. 
  * Ideal use case is with a gpu to maximize performance 
  * There are some improvments I made to this code that have not been added to model.py see gridsearch.py @https://github.com/ythackerCS/GridSearch-Container if you want to see the updates 
  * Parts of the scripts within workspace were written with project specificity in mind so please keep that in mind as you use this code 
  * It is recommended that you have some experience working with docker and specficially building containers for xnat for this to work for your use cases 
  * If you just want to use the code for your own work without docker stuff just navigate to workspace copy the python files from it and edit them 
  
## Future: 
   * Updating code with modifications I did for gridsearch. 
   * generalizing code even more 

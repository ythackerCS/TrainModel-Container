# Example python/pyxnat Docker image and XNAT command

##

### Contents

- build.sh 
    
    Builds and tags the Docker image as `xnat/pyxnat-demo:latest`. Customize the tag in this image to fit your specification

- command2label.py
    
    Script used in `build.sh` to store `command.json` as a Docker image label.

- Dockerfile.base

    Used by `build.sh` as a base for the final `Dockerfile`.

- command.json

    XNAT specific format describing the interface between XNAT Container Service and the Docker image.

- sample-code.py

    Sample pyxnat code to demonstrate XNAT host and filesystem access.

- sample-data folder

    Sample project-level directory structure meant to represent a project-level input mount while running this container in XNAT.
    
    
    
### Docker from command line 
The Docker image can be run on the command line, using the provided sample data and settings for a running XNAT host. When running this container via XNAT Container Service, these environment variables are set to real values at container run-time.

The following Docker cli command assumes:
XNAT username is 'admin'
XNAT password is 'admin'
XNAT host is 'localhost'
Project label is 'test'

`docker run -ti --rm  -e XNAT_HOST=http://localhost -e XNAT_PASS=admin -e XNAT_USER=admin -e PROJECT=test -v $PWD/sample-data/:/input  xnat/pyxnat-demo:latest python workspace/sample-code.py`

You can also run the image interactively:

`docker run -ti --rm  -e XNAT_HOST=http://localhost -e XNAT_PASS=admin -e XNAT_USER=admin -e PROJECT=test -v $PWD/sample-data/:/input  xnat/pyxnat-demo:latest bash`

which opens `bash` in the running container. 


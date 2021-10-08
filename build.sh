cp Dockerfile.base Dockerfile && \
./command2label.py ./xnat/command.json >> Dockerfile && \
docker build -t xnat/model-train:latest .
docker tag xnat/model-train:latest registry.nrg.wustl.edu/docker/nrg-repo/yash/model-train:latest
rm Dockerfile

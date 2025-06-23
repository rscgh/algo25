#!/bin/sh

# Script to build & deploy your docker image
docker build -t eidos-service.di.unito.it/barbano/algonauts:latest . -f Dockerfile.python
docker push eidos-service.di.unito.it/barbano/algonauts:latest

docker build -t eidos-service.di.unito.it/barbano/algonauts:sweep . -f Dockerfile.sweep
docker push eidos-service.di.unito.it/barbano/algonauts:sweep
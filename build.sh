#!/bin/sh

# Script to build & deploy your docker image
docker build -t eidos-service.di.unito.it/barbano/algonauts:latest . -f Dockerfile
docker push eidos-service.di.unito.it/barbano/algonauts:latest
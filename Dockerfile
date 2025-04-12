FROM eidos-service.di.unito.it/eidos-base-pytorch:2.2.1

COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

ENTRYPOINT ["python3"]
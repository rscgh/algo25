FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Add the eidoslab group to the image
# not sure it is really needed but ok
RUN addgroup --gid 1337 eidoslab

# This is wandb stuff
RUN mkdir /.config
RUN chmod 775 /.config
RUN chown -R :1337 /.config

RUN mkdir -p /.local/share/wandb/artifacts/staging
RUN chmod 775 -R /.local/share/wandb/artifacts/staging
RUN chown -R :1337 /.local/share/wandb

# For pytorch checkpoints
RUN mkdir /.cache
RUN chmod 775 /.cache
RUN chown -R :1337 /.cache

# General purpose /data folder
RUN mkdir /data
RUN chmod 775 /data
RUN chown -R :1337 /data

RUN pip3 install transformers
RUN pip3 install torchcodec
RUN pip3 install h5py nilearn nibabel
RUN pip3 install tensorboard

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
RUN apt-get install -y libavutil-dev

RUN pip3 install wandb matplotlib scikit-learn

RUN pip3 freeze > pip-freeze.txt

COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

WORKDIR /src

ENTRYPOINT ["python3"]
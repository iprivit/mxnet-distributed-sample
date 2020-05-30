FROM 763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.6.0-gpu-py27-cu101-ubuntu16.04
LABEL author="vadimd@amazon.com"

COPY container_training /opt/ml/code
WORKDIR /opt/ml/code

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM hvd_launcher.py

RUN pip install gluoncv

# OpenSHH config for MPI communication
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
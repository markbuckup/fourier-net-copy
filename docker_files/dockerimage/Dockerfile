FROM kunalg106/ngp

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install git+https://github.com/kornia/kornia
RUN pip uninstall -y torch
RUN pip uninstall -y torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install pynufft
RUN pip install torchkbnufft
RUN pip install nibabel
RUN pip install av
RUN pip install neptune
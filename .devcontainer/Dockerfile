FROM nvcr.io/nvdlfwea/jax/jax:23.02-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        libomp-dev \
        ffmpeg \
        libjpeg-turbo8 \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libblosc-dev \
        libbrotli-dev \ 
        bzip2 \
        libc-ares-dev \
        libcfitsio-dev \
        dbus \
        expat \
        fontconfig \
        fonttools \
        # freetype \
        libicu-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
        giflib-tools \
        libjpeg-dev \
        liblcms2-dev \
        # liblerc-dev \
        libaec-dev \
        libblas-dev \
        # libcblas-dev \
        # libcurl-dev \
        libdeflate-tools \
        libedit-dev \
        libev-dev \
        libffi-dev \
        libgcc-7-dev \
        libgfortran4 \
        libnghttp2-dev \
        libpng-dev \
        zlib1g-dev \
        libssh2-1-dev \
        libstdc++6 \
        libtiff-dev \
        uuid-dev \
        libwebp-dev \
        libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev \
        libxml2-dev \
        libzopfli-dev \
        lz4 \
        intel-mkl \
        libncurses5-dev libncursesw5-dev \
        #openjpeg-tools \
        libssl-dev \ 
        qt5-default \
        libreadline-dev \
        sqlite3 \
        python3-tk \
        xorg \
        openbox \
        xz-utils \
        python-yaml \
        zlib1g-dev \
        zstd

RUN python3 -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118


RUN python3 -m  pip install \
    numba==0.56.4+1.g772622d0d\
    # cupy \
    black==23.1.0\
    pandas==1.5.2 \
    numpy==1.22.2 \
    comet-ml==3.32.1 \
    h5py==3.6.0 \
    SimpleITK==2.2.1\
    batchgenerators==0.24 \
    dask==2022.11.1 \
    nibabel==5.0.0\
    # intensity-normalization[ants] \
    warp-lang \
    report_guided_annotation==0.2.7\
    ott-jax \ 
    evojax[extra] \
    jraph \
    flax==0.6.7 \
    toolz==0.12.0\
    ipyparallel==8.4.1\
    optuna==3.1.0\
    # equinox \
    einops==0.6.0\
    torchio==0.18.87 \
    jax-triton==0.1.2 \
    opencv-python==4.7.0.68 \
    dm-pix==0.3.4 \
    visu3d==1.4.0 \
    ml-collections==0.1.1 \
    #https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode
    tensorboard-plugin-profile \
    jax-smi==1.0.2 \
    tensorly==0.8.0 \
    tqdm==4.64.1 \
    kymatio==0.3.0 \
    jaxwt \
    matplotlib \
    PyWavelets==1.4.1 \
    Pillow==9.4.0 \
    scikit-image==0.20.0 \
    imageio==2.26.0 \
    jax-cosmo==0.1.0 \
    more-itertools



RUN python3 -m  pip install monai-weekly



RUN ldconfig
# RUN apt-get install -q -y vim nano 

#install go for jax memory profiling


ENV GO_VERSION=1.19.4

RUN apt-get update
RUN apt-get install -y wget git gcc

RUN wget -P /tmp "https://dl.google.com/go/go${GO_VERSION}.linux-amd64.tar.gz"

RUN tar -C /usr/local -xzf "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"
RUN rm "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"


RUN apt install graphviz
RUN go install github.com/google/pprof@latest
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.87


ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/NsightSystems-cli-2022.5.1/target-linux-x64:$LD_LIBRARY_PATH

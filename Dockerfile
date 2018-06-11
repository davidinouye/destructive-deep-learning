FROM ubuntu

WORKDIR /
RUN apt-get update && apt-get install -y \
        wget \
        vim \
        bzip2 \
        git \
        doxygen \
        txt2man \
        pkg-config \
        build-essential \
        cmake \
        liblapack-dev \
        libblas-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-test-dev \
        libboost-serialization-dev \
        libarmadillo-dev \
        binutils-dev \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

# Install mlpack from source since need newer version than on apt-get repositories
# Installs to /usr/local/include/mlpack, /usr/local/lib/, /usr/local/bin/
# 1ee8268 is most recent commit hash as of 3/19/2018 since templates are only 
#  in master rather than released version
RUN git clone https://github.com/mlpack/mlpack.git \
        && cd mlpack \
        && git checkout 1ee8268 \
        && mkdir build \
        && cd build \
        && cmake -Wno-dev ../ \
        && make \
        && make install \
        && make clean \
        && cd ../../ \
        && rm -rf mlpack

# Save linker library path
ENV LD_LIBRARY_PATH /usr/local/lib/:\$LD_LIBRARY_PATH

# Install anaconda
ENV ANACONDA_DIR /anaconda
RUN wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh \
        && /bin/bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p $ANACONDA_DIR \
        && rm Anaconda3-5.1.0-Linux-x86_64.sh
ENV PATH $ANACONDA_DIR/bin:$PATH

# Install anaconda packages
RUN conda update -y -n base conda \
        && conda install -y scikit-learn numpy scipy seaborn matplotlib plotly cython \
        && conda install -y -c conda-forge pot \
        && yes | pip install anytree

# Install python 2 kernel environment and required packages for theano/MAF
SHELL ["/bin/bash", "-c"]
RUN conda create -y -n python2 python=2 ipykernel \
        && source activate python2 \
        && python -m ipykernel install --user \
        && conda install -y pygpu theano \
        && conda install -y plotly cython h5py \
        && source deactivate # Go back to original python 3 environment

# Don't know if this is needed but keeping just in case
ENV MKL_THREADING_LAYER GNU

# From http://singularity.lbl.gov/docs-docker#best-practices
# Must reload linker configuration cache otherwise cannot find shared library when
#  importing into singularity via singularity pull docker://...
RUN ldconfig

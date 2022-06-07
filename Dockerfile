FROM continuumio/miniconda3

RUN git clone https://github.com/sai202022/mle-training.git

COPY deploy/conda/linux_cpu_py39.yml env.yml

RUN conda install -c conda-forge backports.lzma

RUN conda env create -n housing -f env.yml

RUN cd mle-training \
    && conda run -n housing python3 setup.py install\
    && cd src/housing\
    && conda run -n housing python3 ingest_data.py\
    && conda run -n housing python3 train.py \
    && conda run -n housing python3 score.py
    
RUN cd mle-training/tests/unit_tests\
    && conda run -n housing python3 ingest_data_test.py\
    && conda run -n housing python3 train_test.py \
    && conda run -n housing python3 score_test.py

RUN cd mle-training\
    && conda run -n housing pytest tests/functional_tests
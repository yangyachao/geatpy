# Runtime for the official geatpy 2.7.0 binaries (CPython 3.6, x86_64 Linux).
# libgomp is taken from a manylinux wheel because Debian bullseye mirrors are frequently unreachable.
FROM --platform=linux/amd64 python:3.6-slim
RUN pip install -q numpy==1.19.5 scipy==1.5.4 matplotlib==3.3.4 scikit-learn==0.24.2 \
 && mkdir -p /opt/gomp \
 && cp /usr/local/lib/python3.6/site-packages/scikit_learn.libs/libgomp-*.so.1* /opt/gomp/libgomp.so.1 \
 && pip uninstall -y -q scikit-learn
ENV LD_LIBRARY_PATH=/opt/gomp MPLBACKEND=Agg

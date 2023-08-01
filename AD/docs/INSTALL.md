# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04)
* Python 3.6+
* PyTorch 1.7.0, PyTorch 1.8.0, PyTorch 1.8.1
* CUDA 11.1
* [`spconv v2.x`](https://github.com/traveller59/spconv)
* gcc version >= 5.4.0


### Install Autonomous Driving support for Meta-Transformer

a. Clone this repository.
```shell
git clone https://github.com/invictus717/MetaTransformer.git
cd AD
```

b. Install the dependent libraries as follows:

* Install the python dependent libraries.
  ```shell
    pip install -r requirements.txt 
  ```

* Install the gcc library, we use the gcc-5.4 version

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * It is recommended that you should install the latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
    * Also, you should choice **the right version of spconv**, according to **your CUDA version**. For example, for CUDA 11.1, pip install spconv-cu111
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

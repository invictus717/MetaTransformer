# PyTorch Wrapper for Point-cloud Earth-Mover-Distance (EMD)

## Dependency

The code has been tested on Ubuntu 16.04, PyTorch 1.1.0, CUDA 9.0.

## Usage

First compile using
        
        python setup.py install

Then, copy the lib file out to the main directory,

        cp build/lib.linux-x86_64-3.6/emd_cuda.cpython-36m-x86_64-linux-gnu.so .

Then, you can use it by simply

        from emd import earth_mover_distance
        d = earth_mover_distance(p1, p2, transpose=False)  # p1: B x N1 x 3, p2: B x N2 x 3

Check `test_emd_loss.py` for example.

## Author

The cuda code is originally written by Haoqiang Fan. The PyTorch wrapper is written by Kaichun Mo. Also, Jiayuan Gu provided helps.

## License

MIT


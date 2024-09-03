# PyTorch fpsample

PyTorch efficient farthest point sampling (FPS) implementation, adopted from [fpsample](https://github.com/leonardodalinky/fpsample).

**Currently, this project is under heavy development and not ready for production use. The expected release date is before the end of 2024.**

Any contribution is welcome.

## Installation

```bash
# Install from github
pip install git+https://github.com/leonardodalinky/pytorch_fpsample

# Build locally
pip install .
```

## Usage

```python
import torch_fpsample

x = torch.rand(64, 2048, 3)
sampled_points, indices = torch_fpsample.sample(x, 1024)

> sampled_points.size(), indices.size()
Size([64, 1024, 3]), Size([64, 1024])
```

## Reference
Bucket-based farthest point sampling (QuickFPS) is proposed in the following paper. The implementation is based on the author's Repo ([CPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU) & [GPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_GPU)).
```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great works.
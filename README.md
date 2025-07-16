# ComfuUI-RomanticQq
自定义的comfyui插件

## 安装goundingdino
1. 将[groundingdino](https://github.com/IDEA-Research/GroundingDINO.git)项目克隆到ComfyUI-RomanticQq目录下
2. 进入groundingdino目录后执行以下命令
   ```shell
   export CUDA_HOME=/usr/local/cuda
   pip install -e .
3. 在groundingdino目录下创建weights，将权重放在weights目录下，权重链接[groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

## 安装sam2

1. 将[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2.git)项目克隆到ComfyUI-RomanticQq目录下
2. 进入Grounded-SAM-2目录后执行以下命令
   ```shell
   export CUDA_HOME=/usr/local/cuda
   pip install -e .
3. 下载sam权重，将权重放在ComfyUI-RomanticQq/Grounded-SAM-2/checkpoints/目录下，权重链接[sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)
4. 下载groundingdino权重，将权重放在ComfyUI-RomanticQq/Grounded-SAM-2/gdino_checkpoints/目录下，权重链接[groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

## fluxgym_caption
1. 在使用该节点时需要能够访问网络能够访问huggingface，或者使用镜像网站；
2. 设置镜像网站命令
   ```shell
   echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
   source ~/.bashrc

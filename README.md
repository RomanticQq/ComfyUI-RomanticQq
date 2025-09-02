# ComfuUI-RomanticQq
自定义的comfyui插件，该项目是为了方便日常工作，非API结点可以任意使用。

## fluxgym_caption
1. 在使用该节点时需要能够访问网络能够访问huggingface，或者使用镜像网站；
2. 设置镜像网站命令
   ```shell
   echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.bashrc
   source ~/.bashrc

## text_translation
1. 该结点来源于[ComfyUI_Text_Translation](https://github.com/TFL-TFL/ComfyUI_Text_Translation)中的Text_Translation_V2_Full结点，并对其进行重构得到的。因为在使用源结点的过程中出现了一些问题，例如在通过comfyui导出api格式的json文件时，部分连线丢失。

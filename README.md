# facechain-win-test支持linux和windows环境

## conda虚拟环境安装 参考README_ZH.md
```shell
python: py3.8, py3.10
pytorch: torch2.0.0, torch2.0.1
CUDA: 11.7
CUDNN: 8+
OS: Ubuntu 20.04, CentOS 7.9
GPU: Nvidia-A10 24G


conda create -n facechain python=3.8    # 已验证环境：3.8 和 3.10
conda activate facechain

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/facechain.git --depth 1
cd facechain


pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117

pip3 install -r requirements.txt
pip3 install -U openmim
mim install mmcv-full==1.7.0
windows下使用pip install mmcv-full安装

关于sadTalker部分环境参考doc下的installation_for_talkinghead.md
```

## 界面运行
```shell
python3 app.py
Running on public URL: https://cb5bb00f75a086d118.gradio.live
将URL地址在本地电脑浏览器中打开

from facechain.inference import data_process_fn
ModuleNotFoundError: No module named 'facechain'

linux下运行 如果将图片前处理和基模的模型保存到指定目录，在app.py中加入import os  os.environ['MODELSCOPE_CACHE']='./facechain_models'  默认保存在/home/wyw/.cache/modelscope
set PYTHONPATH=.
python3 app.py


windows下运行  如果将图片前处理和基模的模型保存到指定目录，在app-win.py中加入import os  os.environ['MODELSCOPE_CACHE']='G:\\facechain_models' 默认保存在C:\Users\wyw\.cache\modelscope
set PYTHONPATH=.
python3 app-win.py
windows下项目放在E:\PyCharmProject\facechain目录
windows下运行修改app.py中105行'--resume_from_checkpoint="fromfacecommon"'去掉"fromfacecommon"双引号'--resume_from_checkpoint=fromfacecommon'见app-win.py文件，
以及一些windows下路径设置/tmp换成E:\\PyCharmProject\\facechain\\tmp
windows下保存图片，路径中含有中文，则不能顺利的保存和读取图片信息


生成图片结果都保存在目录E:\PyCharmProject\facechain\tmp\qw\inference_result\ly261666\cv_portrait_model\下
Lora模型在E:\PyCharmProject\facechain\tmp\qw\ly261666\cv_portrait_model下

固定模版形象写真，运行出错误时，多试两次
libpng error: Read Error
concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "D:\soft\Miniconda\envs\facechain\lib\concurrent\futures\process.py", line 239, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "E:\PyCharmProject\facechain\facechain\inference_inpaint.py", line 714, in __call__
    h,w,_ = inpaint_img_large.shape
AttributeError: 'NoneType' object has no attribute 'shape'


concurrent.futures.process._RemoteTraceback:
"""
Traceback (most recent call last):
  File "D:\soft\Miniconda\envs\facechain\lib\concurrent\futures\process.py", line 239, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
  File "E:\PyCharmProject\facechain\facechain\inference_inpaint.py", line 760, in __call__
    inpaint_img_rst[cy-cropup:cy+cropbo, cx-crople:cx+cropri] = rst_crop
IndexError: too many indices for array: array is 0-dimensional, but 2 were indexed
```
## Input Image
![aa](https://github.com/Aruen24/facechain-win-test/assets/27750891/a0235cbb-4a04-4de0-8d8b-2a061a7da547)


## 无线风格形象写真
![bb](https://github.com/Aruen24/facechain-win-test/assets/27750891/4863262f-e3c5-4852-a90c-e97ca67cf5a8)


## 固定模版形象写真
![cc](https://github.com/Aruen24/facechain-win-test/assets/27750891/2c08a9a5-8d13-4b6e-a388-bec18d7d18cc)



## 脚本运行(以下部分只在linux下测试过，windows未测试)
```shell
PYTHONPATH=. sh train_lora_test.sh "ly261666/cv_portrait_model" "v2.0" "film/film" "./imgs" "./processed" "./output"

RuntimeError: CUDA error: invalid device ordinal
torch.distributed.elastic.multiprocessing.errors.ChildFailedError

主要是 accelerate's error,解决办法，修改train_lora.sh文件成train_lora_test.sh 去掉 accelerate
将train_lora.sh中accelerate launch facechain/train_text_to_image_lora.py改成python facechain/train_text_to_image_lora.py
```

## 推理
```shell
python run_inference.py
```



## 更多可玩的操作
```shell
# 更多写真风格，可以修改风格Lora模型   原有的是证件照风格
inference.py文件中107行main_model_inference函数中

# 修改写真图片具体内容，如服装物品等
修改inference.py文件中39行main_diffusion_inference函数中的正向和负向提示词

# 后处理阶段进一步调整人脸的细节
修改inference.py文件中139行face_swap_fn函数中人脸融合模型
```
# Open-GroundingDINO微调实战详细教程

 ![](https://i-blog.csdnimg.cn/devpress/blog/fa5b7f5f965f4367ade5bccd3aa40c8b.png) 魔乐社区 文章已被社区收录

加入社区

该文章已生成可运行项目，预览并下载项目源码

## 简介

Grounding DINO是一个强大的**开集目标检测**模型。它最大的特点是能够根据输入的文本描述，在图片中检测出任何你想要寻找的物体，即使这些物体类别没有在训练数据中明确出现过。

![](https://i-blog.csdnimg.cn/direct/869996e1e844434b9d0aa63eea92b99c.png)

**原文链接：[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO "GroundingDINO")**

由于原作只公布了推理代码没有公布train代码，想要做微调需要通过以下GitHub项目：

**Open-GroundingDINO：**[Open-GroundingDino](https://github.com/longzw1997/Open-GroundingDino "Open-GroundingDino")

## 

## 一、配置环境

Open- GroundingDINO 推荐环境：Python 3.7.11, PyTorch 1.11.0, and CUDA 11.3

我使用的系统是 WSL Ubuntu 24.04, Python 3.7, PyTorch 1.11.0, CUDA 11.3

> 评论区@Mr.韩老魔：如果CUDA版本为12.X， 不用降级到CUDA11.3， 实测CUDA12.3安装torch=2.1.0、torchaudio=2.1.0、torchvision=0.16.0、python=3.8.20 也能正常训练，torch=2.1.0支持CUDA11.8和CUDA12.1（即CUDA11和CUDA12理论上都支持），要求python版本最低3.8，3.8版本我这边测试通过了，高于3.8的版本还没试过，这样搭建一个环境能训练也能推理。

创建环境以及安装pytorch（提前安装 anaconda 以及cuda）：

```bash
conda create -n GroundingDINO_train python=3.7 -y
conda activate GroundingDINO_train




运行项目并下载源码bash
```

```bash
`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`

运行项目并下载源码bash
```

克隆 Open-GroundingDINO 仓库:

```bash
`git clone https://github.com/longzw1997/Open-GroundingDino.git && cd Open-GroundingDino/`

运行项目并下载源码bash
```

安装所需依赖项：

```bash
pip install -r requirements.txt 
cd models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../..




运行项目并下载源码bash
```

这里安装成功的话会返回一些 True

下载预训练模型和bert权重：

1\. 以下两种GroundingDINO官方提供的预训练模型权重（我使用的是Swin-T)

| backbone        | Data   | box AP on COCO                                                  | Checkpoint                               | Config                                                                                                                                      |
| --------------- | ------ | --------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| GroundingDINO-T | Swin-T | O365,GoldG,Cap4M                                                | 48.4 (zero-shot)<br><br>57.2 (fine-tune) | [GitHub link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth "GitHub link")      |
| GroundingDINO-B | Swin-B | COCO,O365,GoldG,<br><br>Cap4M,OpenImage,<br><br>ODinW35,RefCOCO | 56.7                                     | [GitHub link](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth "GitHub link") |

2\. bert权重：

[bert权重下载链接](https://huggingface.co/google-bert/bert-base-uncased "bert权重下载链接")

下载以下三个文件，第一个是配置文件，第二个是模型文件，第三个是词汇表

![](https://i-blog.csdnimg.cn/direct/56e5efe83bcf4ba98cf337f22994b8ea.png)

下载好后创建一个新的文件夹bert-base-uncased，将这三个文件打包放入文件夹种

3\. 创建weights文件夹，将上面下载好的文件拷贝到文件夹中，目前文件结构如下：

> Open-GroundingDino/
> 
> `├──`weights/
> 
> `│   ├──`bert-base-uncased/
> 
> `│   │    ├──`config.json
> 
> `│   │    ├──`pytorch_model.bin
> 
> `│   │    └──`cab.txt
> 
> `│   └──`groundingdino_swint_ogc.pth
> 
> `├──` ......
> 
> `└──` ......

## 二、 数据集 格式

**训练的数据集使用的是odvg格式，而验证集使用的是coco格式**

> 评论区@Mr.韩老魔：**对于验证集精度为0**，把val.json文件"annotations"字段里的"category_id"改为从0开始后，"categories"字段里的"id"也要改为从0开始，这两个是对应的，coco默认都是从1开始的。

1\. 使用官方提供的格式转换工具将coco格式转换为odvg格式：

[转换工具链接](https://github.com/longzw1997/Open-GroundingDino/tree/main/tools "转换工具链接")

![](https://i-blog.csdnimg.cn/direct/62b4269a589e4b9d80eab934ba332bb0.png)

转换完成后会得到一个 .jsonl 文件：

![](https://i-blog.csdnimg.cn/direct/db3f19595236417290e2d188b78307fc.png)

示例：

```python
# For OD
{"filename": "000000391895.jpg", "height": 360, "width": 640, "detection": {"instances": [{"bbox": [359.17, 146.17, 471.62, 359.74], "label": 3, "category": "motorcycle"}, {"bbox": [339.88, 22.16, 493.76, 322.89], "label": 0, "category": "person"}, {"bbox": [471.64, 172.82, 507.56, 220.92], "label": 0, "category": "person"}, {"bbox": [486.01, 183.31, 516.64, 218.29], "label": 1, "category": "bicycle"}]}}
{"filename": "000000522418.jpg", "height": 480, "width": 640, "detection": {"instances": [{"bbox": [382.48, 0.0, 639.28, 474.31], "label": 0, "category": "person"}, {"bbox": [234.06, 406.61, 454.0, 449.28], "label": 43, "category": "knife"}, {"bbox": [0.0, 316.04, 406.65, 473.53], "label": 55, "category": "cake"}, {"bbox": [305.45, 172.05, 362.81, 249.35], "label": 71, "category": "sink"}]}}

# For VG
{"filename": "014127544.jpg", "height": 400, "width": 600, "grounding": {"caption": "Homemade Raw Organic Cream Cheese for less than half the price of store bought! It's super easy and only takes 2 ingredients!", "regions": [{"bbox": [5.98, 2.91, 599.5, 396.55], "phrase": "Homemade Raw Organic Cream Cheese"}]}}
{"filename": "012378809.jpg", "height": 252, "width": 450, "grounding": {"caption": "naive : Heart graphics in a notebook background", "regions": [{"bbox": [93.8, 47.59, 126.19, 77.01], "phrase": "Heart graphics"}, {"bbox": [2.49, 1.44, 448.74, 251.1], "phrase": "a notebook background"}]}}




运行项目并下载源码python



运行
```

2\. 创建 **my_label_map.json** 文件，格式如下（标号从0开始，替换为自己的类别）：

```python
`{"0": "person", "1": "bicycle", "2": "car", "3": "motorcycle", "4": "airplane", "5": "bus", "6": "train", "7": "truck", "8": "boat", "9": "traffic light", "10": "fire hydrant", "11": "stop sign", "12": "parking meter", "13": "bench", "14": "bird", "15": "cat", "16": "dog", "17": "horse", "18": "sheep", "19": "cow", "20": "elephant", "21": "bear", "22": "zebra", "23": "giraffe", "24": "backpack", "25": "umbrella", "26": "handbag", "27": "tie", "28": "suitcase", "29": "frisbee", "30": "skis", "31": "snowboard", "32": "sports ball", "33": "kite", "34": "baseball bat", "35": "baseball glove", "36": "skateboard", "37": "surfboard", "38": "tennis racket", "39": "bottle", "40": "wine glass", "41": "cup", "42": "fork", "43": "knife", "44": "spoon", "45": "bowl", "46": "banana", "47": "apple", "48": "sandwich", "49": "orange", "50": "broccoli", "51": "carrot", "52": "hot dog", "53": "pizza", "54": "donut", "55": "cake", "56": "chair", "57": "couch", "58": "potted plant", "59": "bed", "60": "dining table", "61": "toilet", "62": "tv", "63": "laptop", "64": "mouse", "65": "remote", "66": "keyboard", "67": "cell phone", "68": "microwave", "69": "oven", "70": "toaster", "71": "sink", "72": "refrigerator", "73": "book", "74": "clock", "75": "vase", "76": "scissors", "77": "teddy bear", "78": "hair drier", "79": "toothbrush"}`

运行项目并下载源码python



运行
```

3\. 在 **Open-GroundingDino/** 下创建自己的数据集文件夹 **my_finetune_data/** 文件结构如下：

> Open-GroundingDino/
> 
> `└──`my_finetune_data/
> 
> `│   ├──`annotations/
> 
> `│   │    ├──`my_label_map.json       
> 
> `│   │    ├──`train.jsonl                       #使用coco2odvg转换后的train.jsonl
> 
> `│   │    └──`val.json                          #coco格式的val.json
> 
> `│   ├──`train_images/
> 
> `│   │    └──...            `            #训练集图片
> 
> `│   └──` val_images/
> 
> `│         └──...            `            #验证集图片
> 
> `├──` ......
> 
> `└──` ......

4\. 在**Open-GroundingDino/config**路径下找到 **cfg_odvg.py** 和 **datasets_mixed_odvg.json** 进行修改： 

a. 在**cfg_odvg.py**中第38行将text_encoder_type = "bert-base-uncased" 中引号部分替换为刚刚下载的bert权重文件夹的路径

![](https://i-blog.csdnimg.cn/direct/5e6a417bb8dd480a91de2a3ebc1fb726.png)

在**cfg_odvg.py**中第117行将

```python
`use_coco_eval = True`

运行项目并下载源码python



运行
```

替换为

```python
`use_coco_eval = False`

运行项目并下载源码python



运行
```

在最后一行加入你训练时的标签列表（替换为自己的类别名称）：

```python
`label_list=['type1', 'type2', '...']`

运行项目并下载源码python



运行
```

注：如果之前下载的权重文件为Swin-B，还需将第八行的backbone修改为

```python
`backbone = "swin_B_384_22k"`

运行项目并下载源码python



运行
```

![](https://i-blog.csdnimg.cn/direct/53dfa1ff87d14260b0da03ef81e00570.png)

b.把**datasets_mixed_odvg.json**中的内容替换为（将下面的路径替换为自己的实际路径）：

```python
{
  "train": [
    {
      "root": " /home/zhao/Open-GroundingDino/my_finetune_data/train_images/",
      "anno": "/home/zhao/Open-GroundingDino/my_finetune_data/annotations/train.jsonl",
      "label_map": "/home/zhao/Open-GroundingDino/my_finetune_data/annotations/my_label_map.json",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "/home/zhao/Open-GroundingDino/my_finetune_data/val_images/",
      "anno": "/home/zhao/Open-GroundingDino/my_finetune_data/annotations/val.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}




运行项目并下载源码python



运行



![](https://csdnimg.cn/release/blogv2/dist/pc/img/runCode/icon-arrowwhite.png)
```

## 三、训练

1.修改 **train_dist.sh** 文件：

![](https://i-blog.csdnimg.cn/direct/fab68737241043abab9564cd0094620d.png)

将这两处分别改为第一节中下载好的预训练模型和bert权重的路径

2\. 开始训练

```bash
`bash train_dist.sh  ${GPU_NUM} ${CFG} ${DATASETS} ${OUTPUT_DIR}`

运行项目并下载源码bash
```

我这里使用的是双卡训练，训练命令如下：

```bash
`sh train_dist.sh 2 ./config/cfg_odvg.py ./config/datasets_mixed_odvg.json ./training_output`


运行项目并下载源码bash
```

训练过程：

![](https://i-blog.csdnimg.cn/direct/2dcb8c0478d24e4f8c86c359efb66d5d.png)

训练结果：

![](https://i-blog.csdnimg.cn/direct/f9040768a65442ba8cfbbf72ec86a5e9.png)

## 

## 四、推理

Open-GroundingDINO中推理实际上是Grounding DINO官方的推理代码，需要安装GroundingDINO库，问题是之前配的这个环境可以在Open-GroundingDINO上做训练，但是试了很多CUDA11.3对应的pytorch都不能安装GroundingDINO；要想做推理还得重新配置一个环境。我试了一下将CUDA11.3版本的pytorch换成CUD11.8对应的pytorch就可以安装GroundingDINO

> 评论区@Mr.韩老魔：如果CUDA版本为12.X， 不用降级到CUDA11.3， 实测CUDA12.3安装torch=2.1.0、torchaudio=2.1.0、torchvision=0.16.0、python=3.8.20 也能正常训练，torch=2.1.0支持CUDA11.8和CUDA12.1（即CUDA11和CUDA12理论上都支持），要求python版本最低3.8，3.8版本我这边测试通过了，高于3.8的版本还没试过，这样搭建一个环境能训练也能推理。

&nbsp;**原文链接：**[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO "GroundingDINO")****



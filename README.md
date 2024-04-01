# 《用于高分辨率图像合成的Taming Transformers》
##### CVPR 2021（口头报告） 
![预告图](assets/mountain.jpeg)

[**用于高分辨率图像合成的Taming Transformers**](https://compvis.github.io/taming-transformers/)<br/>
[帕特里克·埃塞尔](https://github.com/pesser) *，
[罗宾·罗姆巴克](https://github.com/rromb) *，
[比约恩·奥默](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
* 同等贡献 

**摘要** 我们通过引入卷积 VQGAN，将卷积方法的效率与Transformer的表现力相结合，该卷积 VQGAN 学习了富含上下文的视觉部分的代码簿，其组合由自回归Transformer进行建模。

![预告图](assets/teaser.png)
[arXiv](https://arxiv.org/abs/2012.09841) | [BibTeX](#bibtex) | [项目页面](https://compvis.github.io/taming-transformers/)

### 新闻
#### 2022 年
- 在我们关于[潜在扩散模型](https://github.com/CompVis/latent-diffusion)的新工作中提供了更多预训练的 VQGAN（例如，仅具有 256 个代码簿条目的 f8 模型）。
- 添加了如论文[使用Transformer的高分辨率复杂场景合成](https://arxiv.org/abs/2105.06458)中所提出的场景合成模型，请[参见此部分](#场景图像合成)。
#### 2021 年
- 感谢[rom1504](https://github.com/rom1504)，现在可以[在自己的数据集上训练 VQGAN](#在自定义数据上训练)。
- 包含了一个量化器的错误修复。为了向后兼容，它默认被禁用（这对应于始终使用 `beta=1.0` 进行训练）。使用量化器配置中的 `legacy=False` 来启用它。感谢[richcmwang](https://github.com/richcmwang) 和[wcshin-git](https://github.com/wcshin-git)!
- 我们的论文有了更新：请参见 https://arxiv.org/abs/2012.09841v3 以及相应的更改日志。
- 添加了一个预训练的、[1.4B 的Transformer模型](https://k00.fr/s511rwcv)，用于类条件 ImageNet 合成，它获得了自回归方法中的最先进 FID 分数，并优于 BigGAN。
- 添加了在[FFHQ](https://k00.fr/yndvfu95)和[ CelebA-HQ](https://k00.fr/2xkmielf)上的无条件模型。
- 通过在自关注操作中缓存键/值，添加了加速采样，在 `scripts/sample_fast.py` 中使用。
- 添加了一个[VQGAN](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/)的检查点，该检查点使用 f8 压缩和 Gumbel-Quantization 进行训练。 也请参见我们更新的[重建笔记本](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb)。
- 我们添加了一个[Colab 笔记本](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb)，它比较了两个 VQGAN 和 OpenAI 的[DALL-E](https://github.com/openai/DALL-E)。也请参见[此部分](#更多资源)。
- 现在我们在[表 1](#预训练模型概述)中包括了预训练模型的概述。我们为[COCO](#coco)和[ADE20k](#ade20k)添加了模型。
- 流式演示现在支持图像完成。
- 现在我们包括了一些 D-RIN 数据集的示例，以便您可以运行[D-RIN 演示](#d-rin)而无需首先准备数据集。
- 您现在可以直接使用我们的[Colab 快速启动笔记本](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/taming-transformers.ipynb)进行采样。

## Requirements
一个合适的[conda](https://conda.io/)环境名为“taming”，可以通过以下方式创建和激活：

```
conda env create -f environment.yaml
conda activate taming
```

## 预训练模型概述
以下表格提供了当前所有可用模型的概述。
FID 分数是使用[torch-fidelity](https://github.com/toshas/torch-fidelity)进行评估的。为了参考，我们还包括了最近发布的[DALL-E](https://github.com/openai/DALL-E)模型的自动编码器的链接。请参见相应的[Colab 笔记本](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb)，以进行重建能力的比较和讨论。

| 数据集 | 与训练的 FID | 与验证的 FID | 链接 | 样本（256x256） | 注释 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| FFHQ（f = 16） | 9.6 | -- | [ffhq_transformer](https://k00.fr/yndvfu95) | [ffhq_samples](https://k00.fr/j626x093) |
| CelebA-HQ（f = 16） | 10.2 | -- | [celebahq_transformer](https://k00.fr/2xkmielf) | [celebahq_samples](https://k00.fr/j626x093) |
| ADE20K（f = 16） | -- | 35.5 | [ade20k_transformer](https://k00.fr/ot46cksa) | [ade20k_samples.zip](https://heibox.uni-heidelberg.de/f/70bb78cbaf844501b8fb/) [2k] | 在验证分割（2k 图像）上进行评估 |
| COCO-Stuff（f = 16） | -- | 20.4 | [coco_transformer](https://k00.fr/2zz6i2ce) | [coco_samples.zip](https://heibox.uni-heidelberg.de/f/a395a9be612f4a7a8054/) [5k] | 在验证分割（5k 图像）上进行评估 |
| ImageNet（cIN）（f = 16） | 15.98/15.78/6.59/5.88/5.20 | -- | [cin_transformer](https://k00.fr/s511rwcv) | [cin_samples](https://k00.fr/j626x093) | 不同的解码超参数 |
| | | | | | |
| FacesHQ（f = 16） | -- | -- | [faceshq_transformer](https://k00.fr/qqfl2do8) |
| S-FLCKR（f = 16） | -- | -- | [sflckr](https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/) |
| D-RIN（f = 16） | -- | -- | [drin_transformer](https://k00.fr/39jcugc5) |
| | | | | | |
| VQGAN ImageNet（f = 16），1024 | 10.54 | 7.94 | [vqgan_imagenet_f16_1024](https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/) | [reconstructions](https://k00.fr/j626x093) | 重建 FID。
| VQGAN ImageNet（f = 16），16384 | 7.41 | 4.98 |[vqgan_imagenet_f16_16384](https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/) | [reconstructions](https://k00.fr/j626x093) | 重建 FID。
| VQGAN OpenImages（f = 8），256 | -- | 1.49 |https://ommer-lab.com/files/latent-diffusion/vq-f8-n256.zip | -- | 重建 FID。可通过[潜在扩散](https://github.com/CompVis/latent-diffusion)获得。
| VQGAN OpenImages（f = 8），16384 | -- | 1.14 |https://ommer-lab.com/files/latent-diffusion/vq-f8.zip | -- | 重建 FID。可通过[潜在扩散](https://github.com/CompVis/latent-diffusion)获得。
| VQGAN OpenImages（f = 8），8192，GumbelQuantization | 3.24 | 1.49 |[vqgan_gumbel_f8](https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/) | -- | 重建 FID。
| | | | | | |
| DALL-E dVAE（f = 8），8192，GumbelQuantization | 33.88 | 32.01 | https://github.com/openai/DALL-E | [reconstructions](https://k00.fr/j626x093) | 重建 FID。


## 运行预训练模型

下面的命令将启动一个支持在不同分辨率和图像完成情况下进行采样的 Streamlit 演示。要运行非交互式版本的采样过程，可以将`streamlit run scripts/sample_conditional.py --`替换为`python scripts/make_samples.py --outdir <写入样本的路径>`，并保留其余的命令行参数。

从无条件或类别条件模型进行采样，可以运行`python scripts/sample_fast.py -r <到配置和检查点的路径>`。我们将在下面描述如何从 ImageNet、FFHQ 和 CelebA-HQ 模型进行采样。

### S-FLCKR
![teaser](assets/sunset_and_ocean.jpg)

您还可以[在 Colab 中运行此模型](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/taming-transformers.ipynb)，其中包括开始采样的所有必要步骤。

下载[2020-11-09T13-31-51_sflckr](https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/)文件夹，并将其放置在`logs`中。然后，运行
```
streamlit run scripts/sample_conditional.py -- -r logs/2020-11-09T13-31-51_sflckr/
```

### ImageNet
![teaser](assets/imagenet.png)

下载[2021-04-03T19-39-50_cin_transformer](https://k00.fr/s511rwcv)文件夹，并将其放置在 logs 中。从类别条件 ImageNet 模型进行采样不需要任何数据准备。要为 ImageNet 的 1000 个类中的每一个生成 50 个样本，k 为 600 用于顶部-k 采样，p 为 0.92 用于核采样，温度 t 为 1.0，运行

```
python scripts/sample_fast.py -r logs/2021-04-03T19-39-50_cin_transformer/ -n 50 -k 600 -t 1.0 -p 0.92 --batch_size 25 
```

要限制模型到某些类，请通过`--classes`参数提供它们，用逗号分隔。例如，要对 50 个*鸵鸟*、*边境牧羊犬*和*威士忌水罐*进行采样，运行

```
python scripts/sample_fast.py -r logs/2021-04-03T19-39-50_cin_transformer/ -n 50 -k 600 -t 1.0 -p 0.92 --batch_size 25 --classes 9,232,901 
```
我们建议对最佳结果进行自回归解码参数（顶部-k、顶部-p 和温度）的实验。

### FFHQ/CelebA-HQ

下载[2021-04-23T18-19-01_ffhq_transformer](https://k00.fr/yndvfu95)和[2021-04-23T18-11-19_celebahq_transformer](https://k00.fr/2xkmielf)文件夹，并将它们放置在 logs 中。
同样，从这些无条件模型进行采样不需要任何数据准备。要生成 50000 个样本，顶部-k 采样的 k 为 250，核采样的 p 为 1.0，温度 t 为 1.0，运行

```
python scripts/sample_fast.py -r logs/2021-04-23T18-19-01_ffhq_transformer/ 
```
用于 FFHQ 

```
python scripts/sample_fast.py -r logs/2021-04-23T18-11-19_celebahq_transformer/ 
```
从 CelebA-HQ 模型进行采样。
对于这两个模型，改变顶部-k/顶部-p 参数进行采样可能是有利的。

### FacesHQ
![teaser](assets/faceshq.jpg)

下载[2020-11-13T21-41-45_faceshq_transformer](https://k00.fr/qqfl2do8)并将其放入`logs`。按照[CelebA-HQ](#celeba-hq)和[FFHQ](#ffhq)的数据准备步骤操作。运行
```
streamlit run scripts/sample_conditional.py -- -r logs/2020-11-13T21-41-45_faceshq_transformer/
```

### D-RIN
![teaser](assets/drin.jpg)

下载[2020-11-20T12-54-32_drin_transformer](https://k00.fr/39jcugc5)并将其放入`logs`。要在存储库中包含的几个示例深度图上运行演示，运行

```
streamlit run scripts/sample_conditional.py -- -r logs/2020-11-20T12-54-32_drin_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.imagenet.DRINExamples}}}"
```

要在完整的验证集上运行演示，首先按照[ImageNet](#imagenet)的数据准备步骤操作，然后运行
```
streamlit run scripts/sample_conditional.py -- -r logs/2020-11-20T12-54-32_drin_transformer/
```

## COCO
下载[2021-01-20T16-04-20_coco_transformer](https://k00.fr/2zz6i2ce)并将其放置在`logs`中。要在存储库中包含的几个示例分割图上运行演示，请运行：

```
streamlit run scripts/sample_conditional.py -- -r logs/2021-01-20T16-04-20_coco_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.coco.Examples}}}"
```

## ADE20k
下载[2020-11-20T21-45-44_ade20k_transformer](https://k00.fr/ot46cksa)并将其放置在`logs`中。要在存储库中包含的几个示例分割图上运行演示，请运行：

```
streamlit run scripts/sample_conditional.py -- -r logs/2020-11-20T21-45-44_ade20k_transformer/ --ignore_base_data data="{target: main.DataModuleFromConfig, params: {batch_size: 1, validation: {target: taming.data.ade20k.Examples}}}"
```

## 场景图像合成
![teaser](assets/scene_images_samples.svg)
基于边界框条件的场景图像生成，如我们在 CVPR2021 AI4CC 研讨会论文[高分辨率复杂场景合成与转换器](https://arxiv.org/abs/2105.06458)（在[研讨会页面](https://visual.cs.brown.edu/workshops/aicc2021/#awards)上查看演讲）中所做的那样。支持 COCO 和开放图像的数据集。

### 训练
首先下载第一阶段模型[COCO-8k-VQGAN](https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1/)用于 COCO 或[COCO/Open-Images-8k-VQGAN](https://heibox.uni-heidelberg.de/f/461d9a9f4fcf48ab84f4/)用于开放图像。
更改`data/coco_scene_images_transformer.yaml`和`data/open_images_scene_images_transformer.yaml`中的`ckpt_path`以指向下载的第一阶段模型。
下载完整的 COCO/OI 数据集并在同一文件中调整`data_path`，除非您已经满足使用提供的用于训练和验证的 100 个文件的需求。

代码可以使用`python main.py --base configs/coco_scene_images_transformer.yaml -t True --gpus 0,`运行，或者`python main.py --base configs/open_images_scene_images_transformer.yaml -t True --gpus 0,`。

### 采样 
按照上述描述训练模型或下载预训练模型：
- [开放图像 10 亿参数模型](https://drive.google.com/file/d/1FEK-Z7hyWJBvFWQF50pzSK9y1W_CJEig/view?usp=sharing)经过 100 个时代的训练。在 256x256 像素上，FID 41.48±0.21，SceneFID 14.60±0.15，Inception Score 18.47±0.27。该模型经过 2D 作物的训练，因此非常适合生成高分辨率图像，例如 512x512。
- [具有 1.25 亿参数的上述模型的蒸馏版本](https://drive.google.com/file/d/1xf89g0mc78J3d8Bx5YhbK4tNRNlOoYaO)可在较小的 GPU 上进行 256x256 像素的采样（采样 256x256 像素，FID 43.07±0.40，SceneFID 15.93±0.19，Inception Score 17.23±0.11）。该模型使用 10%的软损失和 90%的硬损失进行了 60 个时代的训练。
- [COCO 30 个时代](https://heibox.uni-heidelberg.de/f/0d0b2594e9074c7e9a33/)
- [COCO 60 个时代](https://drive.google.com/file/d/1bInd49g2YulTJBjU32Awyt5qnzxxG5U9/)（在`assets/coco_scene_images_training.svg`中找到两个 COCO 版本的模型统计信息）

下载预训练模型时，请记得在`configs/*project.yaml`中更改`ckpt_path`以指向您下载的第一阶段模型（参见->训练）。

场景图像生成可以使用`python scripts/make_scene_samples.py --outdir=/some/outdir -r /path/to/pretrained/model --resolution=512,512`运行。

## 在自定义数据上训练

在自己的数据集上训练有助于获得更好的令牌，从而为您的领域获得更好的图像。
以下是实现此目的的步骤：
1. 使用`conda env create -f environment.yaml`，`conda activate taming`和`pip install -e.`安装存储库。
1. 将您的.jpg 文件放在一个文件夹`your_folder`中。
2. 创建两个文本文件`xx_train.txt`和`xx_test.txt`，分别指向您的训练和测试集中的文件（例如`find $(pwd)/your_folder -name "*.jpg" > train.txt`）。
3. 调整`configs/custom_vqgan.yaml`以指向这两个文件。
4. 运行`python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1`在两个 GPU 上进行训练。使用 `--gpus 0,`（带有尾随逗号）在单个 GPU 上训练。

## 在自定义数据上进行训练

在您自己的数据集上进行训练有利于获得更好的令牌，从而为您的领域获得更好的图像。
要使其工作，需要遵循以下步骤：
1. 安装存储库，使用`conda env create -f environment.yaml`、`conda activate taming`和`pip install -e.`。
1. 将您的.jpg 文件放在一个`your_folder`文件夹中。
2. 创建两个文本文件`xx_train.txt`和`xx_test.txt`，分别指向您的训练和测试集中的文件（例如，使用`find $(pwd)/your_folder -name "*.jpg" > train.txt`）。
3. 调整`configs/custom_vqgan.yaml`以指向这两个文件。
4. 运行`python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1`在两个 GPU 上进行训练。使用 `--gpus 0,`（带有尾随逗号）在单个 GPU 上训练。


## 数据准备

### ImageNet
该代码将在首次使用时尝试通过[学术Torrents](http://academictorrents.com/)下载并准备 ImageNet。然而，由于 ImageNet 相当大，这需要大量的磁盘空间和时间。如果您的磁盘上已经有 ImageNet，您可以通过将数据放入`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/`（默认是`~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`）来加快速度，其中`{split}`是`train`/`validation`之一。它应该具有以下结构：

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├──...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├──...
├──...
```

如果您尚未提取数据，您也可以将`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar`（或它们的符号链接）放入`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/` / `${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`，然后将其提取到上述结构中，而无需再次下载。请注意，这只会在既没有`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/`文件夹也没有`${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/.ready`文件时发生。如果您想强制再次运行数据集准备工作，请删除它们。

然后，您需要使用[MiDaS](https://github.com/intel-isl/MiDaS)准备深度数据。创建一个符号链接`data/imagenet_depth`，指向具有两个子文件夹`train`和`val`的文件夹，每个子文件夹都反映了上述相应 ImageNet 文件夹的结构，并包含每个 ImageNet 的`JPEG`文件的`png`文件。`png`将`float32`深度值编码为 RGBA 图像。我们提供了脚本`scripts/extract_depth.py`来生成此数据。**请注意**，该脚本使用[通过 PyTorch Hub 的 MiDaS](https://pytorch.org/hub/intelisl_midas_v2/)。当我们准备数据时，集线器提供了[MiDaS v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)版本，但现在它提供了 v2.1 版本。我们尚未使用通过 v2.1 获得的深度图测试我们的模型，如果您想确保事情按预期工作，您必须调整脚本以确保它明确使用[v2.0](https://github.com/intel-isl/MiDaS/releases/tag/v2)！

### CelebA-HQ
创建一个符号链接`data/celebahq`，指向包含 CelebA-HQ 的`.npy`文件的文件夹（获取它们的说明可以在[PGGAN 存储库](https://github.com/tkarras/progressive_growing_of_gans)中找到）。

### FFHQ
创建一个符号链接`data/ffhq`，指向从[FFHQ 存储库](https://github.com/NVlabs/ffhq-dataset)获得的`images1024x1024`文件夹。

### S-FLCKR
不幸的是，我们不被允许分发我们为 S-FLCKR 数据集收集的图像，因此只能给出它是如何产生的描述。有许多关于[从网络收集图像](https://github.com/adrianmrit/flickrdatasets)的资源可以开始。我们从[flickr](https://www.flickr.com)（请参阅`data/flickr_tags.txt`以获取用于查找图像的完整标签列表）和各种[subreddits](https://www.reddit.com/r/sfwpornnetwork/wiki/network)（请参阅`data/subreddits.txt`以获取使用的所有子版块）收集了足够大的图像。总的来说，我们收集了 107625 张图像，并将它们随机分成 96861 张训练图像和 10764 张验证图像。然后，我们为每幅图像使用[DeepLab v2](https://arxiv.org/abs/1606.00915)训练在[COCO-Stuff](https://arxiv.org/abs/1612.03716)上的分割掩码。我们使用了[PyTorch 实现](https://github.com/kazuto1011/deeplab-pytorch)，并在`scripts/extract_segmentation.py`中包含了一个用于此过程的示例脚本。

### COCO
创建一个符号链接`data/coco`，包含来自 2017 分割中的图像`train2017`和`val2017`，以及它们的注释在`annotations`中。文件可以从[COCO 网页](https://cocodataset.org/)获得。此外，我们使用[COCO-Stuff](https://github.com/nightrome/cocostuff)的[Stuff+thing PNG 风格注释在 COCO 2017 训练val](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)注释，应放置在`data/cocostuffthings`下。

### ADE20k
创建一个符号链接`data/ade20k_root`，包含[ADEChallengeData2016.zip](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)的内容[来自 MIT 场景解析基准](http://sceneparsing.csail.mit.edu/)。

## 训练模型

### FacesHQ

使用以下命令训练一个 VQGAN：
```
python main.py --base configs/faceshq_vqgan.yaml -t True --gpus 0,
```

然后，调整 `configs/faceshq_transformer.yaml` 中配置键 `model.params.first_stage_config.params.ckpt_path` 的检查点路径（或者下载[2020-11-09T13-33-36_faceshq_vqgan](https://k00.fr/uxy5usa9)并放置在 `logs` 中，这对应于预配置的检查点路径），然后运行
```
python main.py --base configs/faceshq_transformer.yaml -t True --gpus 0,
```

### D-RIN

在 ImageNet 上训练一个 VQGAN，使用：
```
python main.py --base configs/imagenet_vqgan.yaml -t True --gpus 0,
```

或者下载一个预训练的，从 [2020-09-23T17-56-33_imagenet_vqgan](https://k00.fr/u0j2dtac) 并放置在 `logs` 下。如果你训练了自己的，调整 `configs/drin_transformer.yaml` 中配置键 `model.params.first_stage_config.params.ckpt_path` 的路径。

训练一个在 ImageNet 深度图上的 VQGAN，使用：
```
python main.py --base configs/imagenetdepth_vqgan.yaml -t True --gpus 0,
```

或者下载一个预训练的，从 [2020-11-03T15-34-24_imagenetdepth_vqgan](https://k00.fr/55rlxs6i) 并放置在 `logs` 下。如果你训练了自己的，调整 `configs/drin_transformer.yaml` 中配置键 `model.params.cond_stage_config.params.ckpt_path` 的路径。

要训练Transformer，运行：
```
python main.py --base configs/drin_transformer.yaml -t True --gpus 0,
```

## 更多资源

### 比较不同的第一阶段模型
不同第一阶段模型的重建和压缩能力可以在这个[Colab 笔记本](https://colab.research.google.com/github/CompVis/taming-transformers/blob/master/scripts/reconstruction_usage.ipynb)中进行分析。特别是，该笔记本比较了两个具有 f = 16 的下采样因子和 1024 和 16384 的代码本维度的 VQGAN，以及具有 f = 8 和 8192 个代码本条目的 VQGAN 和 OpenAI 的[DALL-E](https://github.com/openai/DALL-E)的离散自动编码器（它具有 f = 8 和 8192 个代码本条目）。
![第一阶段 1](assets/first_stage_squirrels.png)
![第一阶段 2](assets/first_stage_mushrooms.png)

### 其他
- [两分钟论文](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg)的[视频总结](https://www.youtube.com/watch?v=o7dqGcLDf0A&feature=emb_imp_woyt)。
- [Gradient Dude](https://www.youtube.com/c/GradientDude/about)的[视频总结](https://www.youtube.com/watch?v=-wDSDtIAyWQ)。
- [权重和偏差报告](https://wandb.ai/ayush-thakur/taming-transformer/reports/-Overview-Taming-Transformers-for-High-Resolution-Image-Synthesis---Vmlldzo0NjEyMTY)，由[ayulockin](https://github.com/ayulockin)总结该论文。
- [What's AI](https://www.youtube.com/channel/UCUzGQrN-lyyc0BWTYoJM_Sg)的[视频总结](https://www.youtube.com/watch?v=JfUTd8fjtX8&feature=emb_imp_woyt)。
- 如果你想在 Colab 上运行 streamlit 演示，可以看看[ak9250 的笔记本](https://github.com/ak9250/taming-transformers/blob/master/tamingtransformerscolab.ipynb)。

### 通过 CLIP 进行文本到图像的优化
VQGAN 已成功用作由[CLIP](https://github.com/openai/CLIP)模型引导的图像生成器，用于从头开始的纯图像生成和图像到图像的转换。我们推荐以下笔记本/视频/资源：

 - [Advadnouns](https://twitter.com/advadnoun/status/1389316507134357506) 的 Patreon 及其相应的 LatentVision 笔记本：https://www.patreon.com/patronizeme
 - [Rivers Have Wings](https://twitter.com/RiversHaveWings) 的[笔记本]( https://colab.research.google.com/drive/1L8oL-vLJXVcRzCFbPwOoMkPKJ8-aYdPN)。
 - [Dot CSV](https://www.youtube.com/channel/UCy5znSnfMsDwaLlROnZ7Qbg) 的[视频](https://www.youtube.com/watch?v=90QDe6DQXF4&t=12s)解释（西班牙语，但有英文字幕可用）

![txt2img](assets/birddrawnbyachild.png)

文本提示：“由孩子画的鸟”

## 致谢
感谢所有提供他们的代码和模型的人。特别是：

- 我们的 VQGAN 架构受到[去噪扩散概率模型](https://github.com/hojonathanho/diffusion)的启发。
- 非常可定制的Transformer实现[minGPT](https://github.com/karpathy/minGPT)。
- 经典的[PatchGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)和[Learned Perceptual Similarity (LPIPS)](https://github.com/richzhang/PerceptualSimilarity)。

## BibTeX

```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

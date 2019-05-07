# NMT
本项目框架参考于斯坦福大学CS224N课程作业。基本模型是seq2seq+attention。NMT_word和NMT_char分别是`word`级和`char`级的翻译，都是可以直接运行的完整项目。训练集共有216617条，验证集有851条，测试集有8064条。

## NMT_word

### 网络架构图

![](https://github.com/wxzcyy/NMT/blob/master/pictures/word_network.jpg)

### 使用指南

首先生成词典文件`sh run.sh vocab`，然后训练`sh run.sh train`，在`Tesla V100`上总共运行了1.8小时，训练完成后再运行`sh run.sh test`测试`BELU`值，能达到`22.72`。

## NMT_char

### 网络架构图
* 编码阶段
![](https://github.com/wxzcyy/NMT/blob/master/pictures/char_network_encoder.jpg)

* 译码阶段
![](https://github.com/wxzcyy/NMT/blob/master/pictures/char_network_decoder.jpg)

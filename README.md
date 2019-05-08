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

参考文献[1]，在编码器一端，我们使用一个卷积层来训练word的词向量，然后输入到LSTM中。
![](https://github.com/wxzcyy/NMT/blob/master/pictures/char_network_encoder.jpg)

* 译码阶段

译码阶段过程和`word`级大致相同。对于无法译出的单词，即输出为`<unk>`，我们使用基于`char`的译码器来进行翻译。
![](https://github.com/wxzcyy/NMT/blob/master/pictures/char_network_decoder.jpg)
### 使用说明
首先生成词典文件`sh run.sh vocab`，然后训练`sh run.sh train`，在`Tesla V100`上总共运行了1.8小时，训练完成后再运行`sh run.sh test`测试`BELU`值，能达到`24.18`。

## 参考文献
* [1]Character-Aware Neural Language Models
* [2]Highway Networks

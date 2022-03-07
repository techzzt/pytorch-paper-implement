# Pytorch

Image data를 바탕으로 모델을 구현하고 정리합니다. 

베이스가 되는 모델부터 최신 모델까지 구조를 공부하는 것을 목표로 합니다.

## Classification 
+ [VGGNet (2014)](https://arxiv.org/pdf/1409.1556.pdf)
  + Very Deep Convolutional Networks for Large-Scale Image Recognition. Karen Simonyan, Andrew Zisserman  
 
+ [GoogLeNet (2014)](https://arxiv.org/abs/1409.4842)
  + Going Deeper with Convolutions Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich 

+ [ResNet (2015)](https://arxiv.org/abs/1512.03385)
  + Deep Residual Learning for Image Recognition Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun   

+ [DenseNet (2016)](https://arxiv.org/abs/1608.06993)
  + Densely Connected Convolutional Networks Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
  
+ [Xception (2016)](https://arxiv.org/abs/1610.02357)
  + Xception: Deep Learning with Depthwise Separable Convolutions François Chollet

+ [ResNeXt (2017)](https://arxiv.org/abs/1611.05431)
  + Aggregated Residual Transformations for Deep Neural Networks Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
  
## Model Summarize Table

|         ConvNet            | Dataset |   Published In     |
|:--------------------------:|:-------:|:------------------:|
|          VGGNet            |  STL10  |      ICLR2015      |
|        GoogleNet           |  STL10  |      CVPR2015      |
|          ResNet            |  STL10  |      CVPR2015      |
|         DenseNet           |    -    |      ECCV2017      |
|          ResNeXt           | CIFAR10 |      CVPR2017      |

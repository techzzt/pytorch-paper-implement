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
 
+ [SEResNet (2017)](https://arxiv.org/abs/1709.01507)
  + Hu, Jie and Shen, Li and Albanie, Samuel and Sun, Gang and Wu, Enhua


## Generative Model
+ [GAN (2014)](https://arxiv.org/abs/1406.2661)

## Model Summarize Table

|         ConvNet            | Dataset |   Published In     |
|:--------------------------:|:-------:|:------------------:|
|          VGGNet            |  STL10  |      ICLR2015      |
|        GoogleNet           |  STL10  |      CVPR2015      |
|          ResNet            |  STL10  |      CVPR2015      |
|         DenseNet           |    -    |      ECCV2017      |
|          ResNeXt           | CIFAR10 |      CVPR2017      |
|         SEResNet           |    -    |      CVPR2018      |


**Table of Contents**

- [CodeLab](#codelab)
- [Computer Vision](#computer-vision)
- [Natural Language Processing](#natural-language-processing)
- [Tabular Data](#tabular-data)
- [Time-Series](#time-series)
- [Reinforcement Learning](#reinforcement-learning)
- [Audio Data](#audio-data)
- [Multi-modality](#multi-modality)
- [Extra](#extra)
- [Pytorch Accelerator](#pytorch-accelerator)

# Computer Vision

**Classification**

- [Model Soup](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Model_Soup) _[Jaehyuk Heo]_
- [Point cloud classification with PointNet](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Point_cloud_classification/Point_cloud_classification.ipynb) _[Hyeongwon Kang]_
- [Involutional neural networks](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Involutional%20neural%20networks/Involutional%20neural%20networks.ipynb) _[Subin Kim]_
- [Image classification with Vision Transformer](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Image_classification_with_Vision_Transformer/Image%20classification%20with%20Vision%20Transformer.ipynb) _[Jaehyuk Heo]_
- [Video Classification with Transformers](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Video_Classification_with_Transformers/Video_Classification_with_Transformers.ipynb) + [Video Vision Transformer](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Video_Classification_with_Transformers/ViViT.ipynb) _[Hyeongwon Kang]_

**Self-Supervised Learning**

- [Semi-supervised image classification using contrastive pretraining with SimCLR](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Semi-supervised%20image%20classification%20using%20contrastive%20pretraining%20with%20SimCLR) _[Subin Kim]_
- [Self-supervised contrastive learning with SimSiam](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Self-supervised_contrastive_learning_with_SimSiam) _[Jaehyuk Heo]_
- [Supervised Contrastive Learning](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Supervised_contrastive_learning/supervised_contrastive_learning.ipynb) _[Subin Kim]_

**Image Denoising**

- [Convolutional autoencoder for image denoising](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Convolutional%20autoencoder%20for%20image%20denoising) _[Jeongseob Kim]_

**Segmentation**

- [Point cloud segmentation with PointNet](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Point_cloud_segmentation_with_PointNet/Point_cloud_segmentation_with_PointNet.ipynb) _[Hyeongwon Kang]_
- [Image segmentation with a U-Net-like architecture](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Image_Segmentation_Unet_Xception/%5BKJS%5DImage%20segmentation%20with%20a%20U-Net-like%20architecture(torch).ipynb) _[Jeongseob Kim]_

**Object Detection**

- [Object Detection with RetinaNet](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Object_Detection_with_RetinaNet) _[Jaehyuk Heo]_

**Knowledge Distillation**

- [Knowledge Distillation](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Knowledge_Distillation/Knowledge%20Distillation%20HJH.ipynb) _[Jaehyuk Heo]_

**Retrieval**

- [Metric learning for image similarity search](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Metric_Learning_for_Image_Similarity_Search/Metric%20learning%20for%20image%20similarity%20search%20HJH.ipynb) _[Jaehyuk Heo]_
- [Image similarity estimation using a Siamese Network with a triplet loss](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Triplet_loss/triplet_loss.ipynb) _[Yonggi Jeong]_


**OCR**

- [OCR model for reading Captchas](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/OCR_model_for_reading_Captchas/captcha_ocr_KSB.ipynb) _[Subin Kim]_

**Augmentation**

- [RandAugment for Image Classification for Improved Robustness](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Augmentation/RandAugment/RandAugment.ipynb) _[Yonggi Jeong]_
- [CutMix data augmentation for image classification](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Augmentation/CutMix%20data%20augmentation%20for%20image%20classification.ipynb) _[Jaehyuk Heo]_

**Clustering**

- [Semantic Image Clustering](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Semantic_Image_Clustering/image_clustering.ipynb) _[Yonggi Jeong]_

**Depth Estimation**

- [Monocular depth estimation](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Monocular_depth_estimation/Monocular_depth_estimation.ipynb) _[Hyeongwon Kang]_

**Attribution Methods**

- [Grad-CAM class activation visualization](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Grad-CAM_class_activation_visualization/Grad-CAM%20class%20activation%20visualization%20HJH.ipynb) _[Jaehyuk Heo]_
- [Model interpretability with Integrated Gradients](https://github.com/hwk0702/keras2torch/blob/main/Computer_Vision/Model_Interpretability_with_Integrated_Gradients/Model%20interpretability%20with%20Integrated%20Gradients.ipynb) _[Jaehyuk Heo]_
- [Visualizing what convnets learn](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Visualizing_what_convnets_learn) _[Jaehyuk Heo]_

**Optimizer**

- [Gradient Centralization for Better Training Performance](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Gradient_Centralization_for_Better_Training_Performance) _[Jaehyuk Heo]_


**Adepter**

- [Finetuning ViT with LoRA](https://github.com/hwk0702/keras2torch/tree/main/Computer_Vision/Finetuning_ViT_with_LoRA) _[Jaehyuk Heo]_

**Generative Models**

- [Variational AutoEncoder](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Variational_AutoEncoder) _[Jaehyuk Heo]_
- [DCGAN to generate face images](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/DCGAN_to_generate_face_images/DCGAN_to_generate_face_images.ipynb) _[Hyeongwon Kang]_
- [Neural style transfer](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Neural_style_transfer/Neural_style_transfer.ipynb) _[Subin Kim]_
- [Deep Dream](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Deep_Dream) _[Jaehyuk Heo]_
- [Conditional GAN](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Conditional_GAN) _[Yonggi Jeong]_
- [CycleGAN](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/CycleGAN) _[Yonggi Jeong]_
- [PixelCNN](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/PixclCNN/pixelcnn.ipynb) _[Jeongseob Kim]_
- [Density estimation using Real NVP](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Normalizing-Flow/RNVP/real-nvp-pytorch.ipynb) _[Jeongseob Kim]_
- [Non-linear Independent Component Estimation (NICE)](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Normalizing-Flow/NICE/NICE_codes.ipynb) _[Jeongseob Kim]_
- [Diffusion generative model(Tutorials)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Tutorial) _[Jeongseob Kim]_
- [Diffusion generative model(Examples - Swiss-roll, MNIST, F-MNIST, CELEBA)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Diffusion) _[Jeongseob Kim]_
- [Score based generative model(Tutorials)](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Score_Diffusion/Tutorial) _[Jeongseob Kim]_


**Adversarial Attacks**

- [Fast Gradient Sign Method](https://github.com/hwk0702/keras2torch/tree/main/Adversarial_Attack/Fast_Gradient_Sign_Method) _[Jaehyuk Heo]_
- [Projected Gradient Descent](https://github.com/hwk0702/keras2torch/tree/main/Adversarial_Attack/Projected_Gradient_Descent) _[Jaehyuk Heo]_

**Adversarial Detection**

- [Detecting Adversarial Examples from Sensitivity Inconsistency of Spatial-Transform Domain](https://github.com/TooTouch/SID) _[Jaehyuk Heo]_

**Anomaly Detection**

- [PatchCore: Towards Total Recall in Industrial Anomaly Detection](https://github.com/hwk0702/keras2torch/tree/main/Anomaly_Detection/PatchCore) _[Jaehyuk Heo]_
- [MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities](https://github.com/TooTouch/MemSeg) _[Jaehyuk Heo]_

# Natural Language Processing


**Classification**

- [Text classification with Switch Transformer](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Text%20classification%20with%20Switch%20Transformer/text_classification_with_switch_transformer.ipynb) _[Subin Kim]_
- [Text classification with Transformer](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Text_Classification_with_Transformers/text_classification_with_transformers_KYK.ipynb) _[Yookyung Kho]_
- [Bidirectional LSTM on IMDB](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Bidirectional_LSTM_on_IMDB/Text_classification_with_bi-LSTM_KJS.ipynb) _[Jeongseob Kim]_

**Generation**

- [Text generation with a miniature GPT](https://github.com/hwk0702/keras2torch/blob/main/Generative_Deep_Learning/Text_generation_with_a_miniauture_GPT/Text_generation_with_a_miniauture_GPT_KSB.ipynb) _[Subin Kim]_
- [Sequence to sequence learning for performing number addition](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Seq2seq_Number_Addition/seq2seq_number_addition_KYK.ipynb) _[Yookyung Kho]_
- [Character-level recurrent sequence-to-sequence model](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Character-level_recurrent_sequence-to-sequence_model/Character_level_Machine_translator_with_seq2seq_KJS_3.ipynb) _[Jeongseob Kim]_
- [English-to-Spanish translation with a sequence-to-sequence Transformer](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Machine_Translation_via_seq2seq_Transformer/spn2eng_Translation_via_seq2seq_Transformer_KYK.ipynb) _[Yookyung Kho]_


**Question Answering**

- [Question Answering with Hugging Face Transformers](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Question_Answering_Huggingface/QA_huggingface_KYK.ipynb) _[Yookyung Kho]_
- [Text Extraction with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Text_Extraction_with_BERT/Text_Extraction_with_BERT_HJH.ipynb) _[Jaehyuk Heo]_


**Pretrained Language Model**

- [End-to-end Masked Language Modeling with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/End-to-end_Masked_Language_Modeling_with_BERT/mlm_and_finetune_with_bert_KSB.ipynb) _[Subin Kim]_
 

**Named Entity Recognition**

- [Named Entity Recognition using Transformers](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Named_Entity_Recognition%20using_Transformers/NER_using_Transformers_KSB.ipynb) _[Subin Kim]_

**Natural Language Inference**

- [Semantic Similarity with BERT](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Semantic_Similarity_with_BERT/Semantic_Similarity_with_BERT_HJH.ipynb) _[Jaehyuk Heo]_


**Table MRC**

- [Table Pre-training with TapasForMaskedLM](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Table_pretraining_with_TAPAS/Table_Pretraining_with_TapasForMaskedLM_KYK.ipynb) _[Yookyung Kho]_


**Tutorial**

- [TorchText introduction](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Extra/TorchText_introduction_KJS.ipynb) _[Jeongseob Kim]_




# Tabular Data

**Classification**

- [Classification with Gated Residual and Variable Selection Networks](https://github.com/hwk0702/keras2torch/tree/main/Structured_Data/Classification_with_Gated_Residual_and_Variable_Selection_Networks) _[Hyeongwon Kang]_
- [Structured data learning with TabTransformer](https://github.com/hwk0702/keras2torch/tree/main/Structured_Data/Structured_data_learning_with_TabTransformer) _[Hyeongwon Kang]_


**Recommendation**

- [Collaborative Filtering for Movie Recommendations](https://github.com/hwk0702/keras2torch/blob/main/Structured_Data/Collaborative_Filtering_for_Movie_Recommendations/Collaborative_Filtering_for_Movie_Recommendations.ipynb) _[Hyeongwon Kang]_
- [A Transformer-based recommendation system](https://github.com/hwk0702/keras2torch/blob/main/Structured_Data/Collaborative_Filtering_for_Movie_Recommendations/Collaborative_Filtering_for_Movie_Recommendations.ipynb) _[Hyeongwon Kang]_

**Anomaly Detection**

- [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://github.com/SunwooKimstar/DAGMM.git) _[Sunwoo Kim]_

# Time-Series

**Anomaly Detection**
- [Timeseries anomaly detection using an Autoencoder](https://github.com/hwk0702/keras2torch/blob/main/Timeseries/Timeseries_anomaly_detection_using_an_Autoencoder/Timeseries_anomaly_detection_using_an_Autoencoder.ipynb) _[Hyeongwon Kang]_
  
**Classification**
- [Timeseries classification with a Transformer model](https://github.com/hwk0702/keras2torch/blob/main/Timeseries/Timeseries_classification_with_a_Transformer_model/Timeseries_classification_with_a_Transformer_model.ipynb) _[Hyeongwon Kang]_

**Forecasting**
- [Timeseries forecasting for weather prediction](https://github.com/hwk0702/keras2torch/tree/main/Timeseries/Timeseries_forecasting_for_weather_prediction) _[Hyeongwon Kang]_


# Reinforcement Learning

- [Actor Critic Method](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Actor_Critic_Method/Actor_Critic_Method_KHW.ipynb) _[Hyeongwon Kang]_
- [Deep Deterministic Policy Gradient (DDPG)](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/DDPG/DDPG.ipynb) _[Hyeongwon Kang]_
- [Deep Q-Learning for Atari Breakout](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Deep_Q_Learning_for_Atari_Breakout/Deep_Q_Learning_for_Atari_Breakout_KHW.ipynb) _[Hyeongwon Kang]_
- [Proximal Policy Optimization](https://github.com/hwk0702/keras2torch/blob/main/Reinforcement_Learning/Proximal_Policy_Optimization/Proximal_Policy_Optimization.ipynb) _[Hyeongwon Kang]_


# Audio Data

**Recognition**

- [Speaker Recognition](https://github.com/hwk0702/keras2torch/blob/main/Audio_Data/Speaker%20Recognition.ipynb) _[Subin Kim]_


# Multi-modality

**Vision-Langauge**

- [Multimodal entailment](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Multimodal_Entailment/multimodal_entailment_KYK.ipynb) _[Yookyung Kho]_
- [Natural language image search with a Dual Encoder](https://github.com/hwk0702/keras2torch/blob/main/Natural_Language_Processing/Natural_language_image%20search_with_a_Dual_Encoder/nl_image_search_KSB.ipynb) _[Subin Kim]_



# Extra

- [Distributions_TFP_Pyro](https://github.com/hwk0702/keras2torch/tree/main/Generative_Deep_Learning/Normalizing-Flow/Framework_practice/Distributions_TFP_Pyro) _[Jeongseob Kim]_



# Pytorch Accelerator

- [Huggingface Accelerator](https://github.com/TooTouch/Pytorch-Accelerator-Test) _[Jaehyuk Heo]_
- [Automatic Mixed Precision](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/AMP) _[Jaehyuk Heo]_ 
- [Gradient Accumulation](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/gradient_accumulation) _[Jaehyuk Heo]_
- [Distributed Data Parallel](https://github.com/hwk0702/keras2torch/tree/main/Pytorch-Accelerator/DDP) _[Jaehyuk Heo]_

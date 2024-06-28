# Dancing in Style: Classifying Dance Videos By Style

<div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
  <div style="flex: 1; text-align: center; margin: 10px;">
    <img src="https://media.giphy.com/media/zvIIy47mvytas2oyUI/giphy.gif" alt="Ballet" style="max-width: 100%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center; margin: 10px;">
    <img src="https://media.giphy.com/media/KRfOTpKsHpXyb9tfwI/giphy.gif" alt="Breakdance" style="max-width: 100%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center; margin: 10px;">
    <img src="https://media.giphy.com/media/ridB5JeMnuK1tAb9ZB/giphy.gif" alt="Zumba" style="max-width: 100%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center; margin: 10px;">
    <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHJvd2docHo4dTV3djN5MWJleG5mN2tlc3RhYmtqb3F0MzZuZ212eSZlcD12MV9pbnRlcm5naWZfYnlfaWQmY3Q9Zw/z7PgsPmsSBYsxAavPJ/giphy-downsized-large.gif" alt="Tango" style="max-width: 100%; height: auto;">
  </div>
</div>

**Authors**:
| Sophie Wu | Han Dao | Nathan Guzman |
| --- | --- | --- |
| Stanford University | Stanford University | Stanford University |

## Abstract
The task of classifying human actions in videos has been a significant area of computer vision research, with dance being a culturally rich and stylistically diverse subset. In this project, we present a novel approach to classify dance videos by style using an enhanced Two-Stream Inflated 3D ConvNet (I3D) model. We leverage the Kinetics-700 and Let's Dance datasets, combining them to create a robust dataset for training and evaluation. Our method and experimentation involve extensive hyperparameter tuning to improve the model performance. Additionally, we experimented with transfer learning by fine-tuning two different existing I3D models, originally based on the ResNet50 and the ResNet50 (NonLocal Dot Product) backbones, on our dance-focused dataset. Experimental results demonstrate the effectiveness of our approach, achieving accuracy in classifying various dance styles.

## Introduction
The task of classifying human action in videos has long since been a major area of computer vision research. We aim to focus on a narrower subset: learning to recognize different styles of dance. Dance is an integral part of cultures around the world, and each style is a product of unique regional, historical, and social influences. Automatically and systematically classifying these different forms of dance would help preserve and study them, potentially revealing new insights into what aspects make each dance style unique.

This project was completed as the final project for our CS231N Deep Learning for Computer Vision class at Stanford University, which we took in Spring 2024.

## Dataset and Features

### Creating the Datasets
To develop a comprehensive dataset for dance video classification, we utilized two primary sources: the Kinetics-700 dataset and the Let's Dance dataset.

1. **Filtering Kinetics-700**: We began by filtering the Kinetics-700 dataset to extract videos specifically related to dance. This involved identifying and selecting videos labeled with dance-related actions. The identified dance categories included 19 unique labels.

2. **Combining with Let's Dance Dataset**: Next, we incorporated the Let's Dance dataset, which contains detailed annotations for various dance styles. We mapped the dance labels from Let's Dance to align with those from Kinetics-700, ensuring consistency across our combined dataset.

3. **Data Augmentation**: To increase the size and variability of our dataset, we applied several data augmentation techniques to the training split. These techniques included horizontal flip, vertical flip, color jitter, and noise addition.

4. **Splitting the Dataset**: The final merged dataset was split into three sets:
   - Training Set (90%)
   - Validation Set (5%)
   - Test Set (5%)

### Annotations
We generated annotation files for each split, listing the video names and their corresponding integer-encoded labels.

### Class Labels
The class labels were mapped to the following dance styles:
- 0: belly dancing
- 1: breakdancing
- 2: country line dancing
- 3: cumbia
- 4: dancing ballet
- 5: dancing charleston
- 6: dancing gangnam style
- 7: dancing macarena
- 8: jumpstyle dancing
- 9: krumping
- 10: moon walking
- 11: mosh pit dancing
- 12: robot dancing
- 13: salsa dancing
- 14: square dancing
- 15: swing dancing
- 16: tango dancing
- 17: tap dancing
- 18: zumba

## Methods

### Overview
Our primary model, the Two-Stream Inflated 3D ConvNet (I3D), builds on state-of-the-art image classification architectures by extending them into the spatio-temporal domain. This section includes the mathematical formulations of our input, output, and loss functions, and details the modifications made to existing models to enhance performance on action recognition tasks.

### Two-Stream Inflated 3D ConvNet (I3D)
The I3D model inflates 2D ConvNet filters and pooling kernels into 3D, enabling the network to learn spatio-temporal features directly from video data. The architecture is based on the Inception-v1 network, which we extend by converting 2D filters $N \times N$ into 3D filters $N \times N \times N$. The model comprises two parallel streams: one for RGB frames and one for optical flow.

<div style="display: flex; justify-content: center; align-items: flex-end; margin-top: 20px;">
  <div style="flex: 1; max-width: 70%; margin: 10px;">
    <img src="https://i.ibb.co/ckV2N43/i3d-architecture.png" alt="I3D Architecture" style="max-width: 100%; height: auto;">
  </div>
</div>

### Training Procedure
We trained the I3D model on the K700-2020 dataset, focusing specifically on dance videos. To implement our training procedures, we built on top of MMAction2, an open-source toolbox for video understanding based on PyTorch.

### Algorithm Steps
1. **Initialization**: Inflate 2D filters from a pre-trained Inception-v1 model into 3D filters.
2. **Data Augmentation**: Apply random cropping, resizing, and flipping to the input frames.
3. **Forward Pass**: Compute the spatio-temporal features using 3D convolutions and pooling.
4. **Loss Computation**: Calculate the cross-entropy loss between the predicted and true labels.
5. **Backpropagation**: Update the network weights using gradient descent with momentum.
6. **Testing**: Evaluate the model on test sets by averaging predictions across video frames.

### Hyperparameter Tuning
We performed hyperparameter tuning using Optuna to identify the optimal learning rate, dropout rate, and optimizer for our model. The best performing hyperparameters were:
- Learning rate: $7.87 \times 10^{-4}$
- Dropout rate: 0.2966
- Optimizer: SGD

### Transfer Learning and Finetuning
We experimented with transfer learning by fine-tuning existing I3D models. Finetuning was performed by loading pretrained weights from two different I3D models (ResNet50 and ResNet50 (NonLocal Dot Product)) and adapting them to our dance-focused dataset.

## Experimental Results

### Evaluation Metrics
We used the following metric to evaluate our model's performance:
- **Accuracy**: The proportion of correct predictions out of the total number of predictions.

### Initial Experiment Results
Initial results showed that the baseline I3D model achieved moderate accuracy in classifying dance styles. The model's performance improved significantly with hyperparameter tuning and transfer learning.

### Hyperparameter Tuning Results
Table 1 summarizes the training and validation losses, as well as the validation accuracy over 10 epochs:

| Epoch | Training Loss | Validation Loss | Validation Accuracy (%) |
|-------|----------------|-----------------|-------------------------|
| 1     | 2.90550        | 2.81120         | 15.32                   |
| 2     | 2.88860        | 2.79430         | 15.47                   |
| 3     | 2.87170        | 2.77740         | 15.41                   |
| 4     | 2.85480        | 2.76050         | 15.48                   |
| 5     | 2.83790        | 2.74360         | 15.44                   |
| 6     | 2.82100        | 2.72670         | 15.45                   |
| 7     | 2.80420        | 2.70980         | 15.46                   |
| 8     | 2.78730        | 2.69300         | 15.43                   |
| 9     | 2.77040        | 2.67600         | 15.49                   |
| 10    | 2.75350        | 2.65910         | 15.45                   |

<div style="display: flex; justify-content: center; align-items: flex-end; margin-top: 20px;">
  <div style="flex: 1; max-width: 75%; margin: 10px;">
    <img src="https://i.ibb.co/YPssRzT/hyper-training-validation-loss.png" alt="Training and Validation Loss" style="max-width: 100%; height: auto;">
  </div>
</div>

### Transfer Learning and Finetuning Results
The best results from fine-tuning the I3D models on our dataset are summarized below:

| Pretrain Model      | Learning Rate | Epochs | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
|---------------------|---------------|--------|---------------------|---------------------|
| ResNet50            | $1 \times 10^{-2}$ | 3      | 51.61               | 82.80               |
| ResNet50 (NonLocal) | Optimal: $7.87 \times 10^{-4}$ | 10     | 56.90               | 86.31               |

## Conclusion and Future Work
In this study, we explored various experimentations, including hyperparameter tuning and transfer learning, to improve the performance of our model. Through extensive experimentation, we identified optimal hyperparameters that significantly enhanced model performance. Our results indicate that transfer learning is effective for this specific domain, and general action recognition models can be fine-tuned to classify and recognize more specific variations of actions, such as dance styles.

### Future Work
Future work includes:
- Implementing additional data augmentation techniques.
- Extending training time to further enhance model performance.
- Conducting a broader hyperparameter search.
- Exploring ensemble methods to improve robustness and accuracy.
- Performing further pretraining and finetuning on different pretrained models.
- Exploring additional regularization techniques to reduce overfitting.

## References
1. Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. CoRR, abs/1705.07750.
2. Carreira, J., Noland, E., Banki-Horvath, A., Hillier, C., & Zisserman, A. (2020). A short note on the kinetics-700-2020 human action dataset. arXiv preprint arXiv:2010.10864.
3. Carreira, J., & Zisserman, A. (2019). Kinetics-700 dataset. arXiv preprint arXiv:1907.06987.
4. Castro, D., Hickson, S., Sangkloy, P., Mittal, B., Dai, S., Hays, J., & Essa, I. A. (2018). Let’s Dance: Learning from Online Dance Videos. CoRR, abs/1801.07388.
5. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). Slowfast networks for video recognition. IEEE International Conference on Computer Vision (ICCV), 6202–6211.
6. Karpathy, A., Toderici, G., Shetty, S., Leung, T., Sukthankar, R., & Fei-Fei, L. (2014). Large-scale video classification with convolutional neural networks. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1725–1732.
7. MMAction2 Contributors. (2020). OpenMMLab’s Next Generation Video Understanding Toolbox and Benchmark. Retrieved from https://github.com/open-mmlab/mmaction2.
8. Simonyan, K., & Zisserman, A. (2014). Two-stream convolutional networks for action recognition in videos. CoRR, abs/1406.2199.
9. Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3d convolutional networks. IEEE International Conference on Computer Vision (ICCV), 4489–4497.
10. Wang, H., & Schmid, C. (2013). Action recognition with improved trajectories. IEEE International Conference on Computer Vision (ICCV), 3551–3558.
11. Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local neural networks. CVPR.

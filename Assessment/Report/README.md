# Face Mask detection 

Xiaohan Feng

Project link：<https://github.com/Dora-fxh/casa0018/tree/main/Assessment/Projects/Final%20Project>


## Introduction
Coronavirus disease 2019 (COVID‐19) spreads widely around the whole world since the end of 2019 (Huang et al., 2020). Many measures have been implemented to curb it, but there are still around 4,550,837 new cases per week in April 2021. According to the Weekly epidemiological update on COVID-19 of the World Health Organization¬—WHO (2021), there have 135,057,587 confirmed cases and 2,919,932 deaths by 13 April 2021. Though many detection and diagnosis methods of COVID-19 has been developed (Bai et al., 2020; Wang et al., 2020), the most effective way to control the epidemic disease nowadays is to wear masks in public, since it can prevent small respiratory droplets from transmitting the virus among people (Howard et al., 2021). Cheng et al. (2020) also verified the contribution of masks in this situation. Most governments around the world encourage and even published temporary legislation to demand citizens to wear face masks in public (Congressional Documents and Publications, 2020). But there are still many people who do not willing or forget to wear face masks when they come into public places. Our project aims to automatically identify whether faces detected are with or without masks. It could be applied in some public places and signal the guards when finding someone who does not abide by the regulations. In this case, guards do not need to patrol around or staring surveillance cameras all the time. A lot of work for guards can be saved and the effectiveness is also improved. 

## Research Question
How to use the webcam to detect whether people wear masks properly and what measures can be taken to improve the accuracy of the face mask detection model which is originally base on the simple [Cats vs Dogs](https://github.com/djdunc/casa0018/blob/main/Week2/CASA0018_2_3_Cats_v_Dogs.ipynb) example?

## Application Overview
Following diagram shows how our application goes. The main loop contains 4 steps. Firstly, the image is captured from the web camera using the OpenCV library. Secondly, the function called CascadeClassifie in OpenCV and the haarcascade_frontalface_alt2.xml file (this file could be found in your cv2 folder when you successfully downloaded the opencv library. I also attached the file in case you cannot find it) is utilized to detect and locate the face. After that, the mask classification model we trained is used to identify whether the face detected has a mask or not. Lastly, if the probability of no mask is larger than 0.5, a no-mask label will show above the face detected.

![overviewDiagram](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/overview.png)

What was talked about above is the final one—all the components were connected to the main loop with the red line, while those connected with black one is other trails we did. At first, the arducam is used in the camera model(the ① in the figure). However, we find the arducam did not work and after the detailed check (the main steps is in following figures) , we consider it’s broken (we checked using the ArduCAM_Mini_2MP_Plus_function. Before uploading, we make sure only uncomment the hardware platform we need in memorysaver.h. After uploading successfully, we double checked the wiring and Arducam pins and Arduino pins are correspondingly connected. Then, we open the app called ArduCam_Host_V2 and comfirms that the port is com5 which is consistent with Arduino IDE). However, if it works, we will not need the face detection part and the final model we need will be tflite file(tflite of int8,uint8 and int16 are produced as we also tried to deploy the model in the andriod APP). Besides, the input image size will also be smaller than our current one (the rows and columns of the image’s pixels will be 96x96 and the number of color channels used to represent each pixel will be 1—grayscale).
For the face detection model, at first, the paddle library is used. However, it’s very complicated when transforming our saved_model into the model that paddle accepted which is in the grey part. Then, we turn to the dlib library, but the effect is still not good as when we put on the mask, it’s hard to detect our face. At last we find the one we actually used. Although it works, it require a computer with High-end configuration. Otherwise, it will be quite slow during detection.

![test_on_arducam](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/test_on_arducam.png)

## Data
The main data set we used is collected by AIZOO and can be downloaded [here](https://drive.google.com/file/d/1QspxOJMDf_rAWVV7AU_Nc0rjo1_EPEDW/view).
It contains two folders, one is for training which has 6120 images (3006 of them come from the MAFA dataset with face masks and 3114 from WIDER Face without masks), the other is as the validation set and has 1839 pictures(contains 799 mask samples and 1040 no-mask samples). Although the dataset with mask and with no masks was mixed together, every image has a corresponding XML file which includes the label, the bounding box of the face, etc. Thus, we use the label in the XML file to divide the data set into what is similar to the ‘Cats vs Dogs’ example. During this process, two images in the training folder are detected with the wrong filename extension(not belong to any suffix names of the image file like webp, jpeg, jpg, and png.), so they were deleted. At the same time, all the figures are filled into squares since directly squeeze them into a square will distort the figures. An example of an original picture and its distorted picture and the filled picture is as follows：

![datafirstprocess](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/datafirstprocess.png)

However, when all the experiment was done, we find that no matter what parameters we adjust, the accuracy is still relatively low (less than 70%). The reason may relate to our dataset. Our model is a classification model, if there are many other distractions in the background (like there are some other people besides our object), the accuracy will be largely influenced. In this case, we decide to crop only the face in the picture by the bounding boxes got from the XML File. Additionally, we realize that although the model might perform well in the existing “perfect” dataset, it might underfit the new data. Thus, we will not fill the cropped data into squares. These three pictures show the initial dataset and what the data processed by two different processing methods like. 
![dataprocess](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/Data%20preprocess.png)


Moreover, when generating data using ImageDataGenerator function, we randomly rescale, flip, shear, zoom, reotate and shift the photo to produce more samples to adapt the changeable environment. The figure below shows how our generator works.

![randomzoom](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/randomzoom.png)


Another dataset which is similar to our cropped dataset and consists of 700 images belonging to two classes(with mask: 350, without mask: 350) from [here](https://github.com/chandrikadeb7/Face-Mask-Detection). It is only used as the testing set to roughly test our model, because in its mask folder, there are some people whose mask did not cover their nose. In our model, we do not want the masks are recognized even if they are not properly worn.

## Model
We use the convolutional neural network (CNN) model to do the project. It is a widely used model in the image recognition field, and there are many face-detection studies based on CNN (Seo and Chung, 2019). We define the input shape to be 150x150 pixels with 3 colour depth and the model we final used has 7 layers with the first 4 being a mix of convolution and pooling. It looks as follows:
![model](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/finalmodel.png)

The role of the convolutional layer is to reduce the images into a form that is easier to process, without losing features that are critical for getting a good prediction and there are 32 and 64 convolutions of size 3x3 for convolution layers separately.
The MaxPooling layer uses the maximum value from the portion of the image to reduce the spatial size of the whole image and thus decrease the computational power (Sumit Saha, 2018). By specifying a size 2x2 we are effectively quartering the size of the image.
Layer 5 is called 'Flatten' which takes the previous layer and turns it into a 1-dimensional set. The 6th layer 'Dense' has 512 neurons and the last layer 'Dense' has 1 neuron. The fully connected layer is used to learn the non-linear combination of the huge-volume features from the convolutional layer and the final classification output (Sumit Saha, 2018). It implements an activation function that looks at all the probabilities in that layer of neurons and sets the highest value to 1 and all the others to 0 - this makes it programmatically easier to find the most likely solution.
For the activation function, 'relu' (Rectified Linear Unit - ReLU) activation function is one of the most commonly activation function used for convolution layer.It converts any negative value to zero and the sigmoid function is often used for binary classifiation because its output range is (0,1).


![activation function](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/activation%20function.png)

What's more We use binary crossentropy as the loss function. It evaluated on the log loss between the predicted values and the ground truth, and because of it, the accuracy in metrics is the binary_accuracy.

![binary](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/binary.png)

where $\hat{y_i}$ is the ith scalar value in the model output, $y_i$ is the corresponding target value, and output size is the number of scalar values in the model output.

Root Mean Square Propagation is used as the optimizer

![optimizer](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/root%20mean%20square.png)

Besides, we set the early stop to help us break the training process. When there are 5 epochs with no improvement, the training will be stopped. Also we monitors the validation accuracy and if no improvement is seen for 2 epochs, the learning rate will be reduced to half of it (no less than 1e-5).

We also tried the model with dropout layer. It can probabilistically remove the inputs of a layer, which makes the network more robust to the inputs and  is often used after the pooling layers or the two dense fully connected layers (Brownlee, 2018). Additionally, the number of convolution layer and pooling layer is also changed to make the model more complex. The reason we do not choose them will be discussed later in the experiments part.

## Experiments
The following figure is the result get from our final model, as we can see, it has no over-fitting problem, so the dropout layer may not be useful. 

![result1](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/2noDropout.png)

The results of adding the dropout layer confirmed our hypothesis. It loses some accuracy and the distance between two lines (training and validation) is not stable as the former one.

![result2](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/2dropout.png)

We also tried 6 layers of convolution layer and pooling layer with and without dropout layer. This time, the latter one performs better. It seems the dropout layer is effective in more complicated models. 

![result3](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/3nodrop.png)
![result4](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/3dropout.png)

The more layers, the more problems that need to be solved, such as the selection of activation function and the number of neurons, and the improvement is very weak, so we decide to keep model simple—with 2 convolution layer. 
Thus, we choose our current model.
The initial learning rate is also changed. We choose four values: 0.00001, 0.0001, 0.001, and 0.01. From the following figures, we can see that when the learning rate is small, the convergence speed is slow and when the learning rate is large, the oscillation does not converge. Setting the learning rate to 0.0001 and 0.001 has a similar result. We finally decide to set it to 0.001 since it processes faster and is more unlikely to fail into local optimality.

![lr](https://github.com/Dora-fxh/casa0018/blob/main/Assessment/Report/figures/chooselr.png)

Then, we use above parameters and model structure we selected to train the model.The model trained is tested in three ways. 
Firstly, We randomly select one of the picture in test set to show the probability and which class it is classfied to see whether it is correct. Secondly, we use the whole test set to evaluate the model and get a loss of 0.1953 and an accuracy of 0.9329.
Lastly, we use the opencv library to call the camera of the computer. our classification model is combined with the face detection model to detect whether the person in front of the camera wears a mask. From the video (test.mp4) we can see that it will not misclassify my hand as the mask and when the mask wears beneath the nose, it can recognize it as no-mask, it's exactly we want.

<video src="https://github.com/Dora-fxh/casa0018/raw/main/Assessment/Report/figures/test.mp4"  controls="controls" width="500" height="300"></video>

## Results and Observations
The model has indeed improved than the model presented in the presentation(both ipynb files are in the guthub link at the top). The reason may due to our data. The data set we choose to train the model have diverse images, and the label it has is accuracy. It is the main reason that we can detected a person who do not wear mask correctly as no-mask. What's more, the accuracy in the test set is still relatively high which indicates that the model we built has no overfitting problem and the generalization ability is good. The reason behind might be what we did in the ImageDataGenerator part. Besides, We found that the dropout layer is more useful when the model is complicated. 
However, there are some limitations in the experiment. When using the webcam to test the model, we can only detect the front face, as the face detection model can only recognize the front face. We will keep finding the suitable model.
In addition, we only change the initial learning rate, other important parameters, such as the batch size is not explored. The optimizer Adam is recognized as a better choice sometimes. Later, it will also be checked. The most important is that we should explore in detail which images are wrongly classified in the test set. When dealing all the limitations, the model might be further improved.


## Bibliography
Bai, H. X., Wang, R., Xiong, Z., Hsieh, B., Chang, K., Halsey, K., Tran, T. M. L., Choi, J. W., Wang, D.-C., Shi, L.-B., Mei, J., Jiang, X.-L., Pan, I., Zeng, Q.-H., Hu, P.-F., Li, Y.-H., Fu, F.-X., Huang, R. Y., Sebro, R., Yu, Q.-Z., Atalay, M. K. and Liao, W.-H. (2020). ‘Artificial Intelligence Augmentation of Radiologist Performance in Distinguishing COVID-19 from Pneumonia of Other Origin at Chest CT’. Radiology, 296 (3), pp. E156–E165. doi: 10.1148/radiol.2020201491.

Brownlee, J. (2018). How to Reduce Overfitting With Dropout Regularization in Keras. Machine Learning Mastery. Available at: https://machinelearningmastery.com/how-to-reduce-overfitting-with-dropout-regularization-in-keras/.

Cheng, V. C.-C., Wong, S.-C., Chuang, V. W.-M., So, S. Y.-C., Chen, J. H.-K., Sridhar, S., To, K. K.-W., Chan, J. F.-W., Hung, I. F.-N., Ho, P.-L. and Yuen, K.-Y. (2020). ‘The role of community-wide wearing of face mask for control of coronavirus disease 2019 (COVID-19) epidemic due to SARS-CoV-2’. Journal of Infection, 81 (1), pp. 107–114. doi: 10.1016/j.jinf.2020.04.024.

Congressional Documents and Publications. (2020). ‘SENATORS MARKEY AND BLUMENTHAL ANNOUNCE NATIONAL FACE MASK MANDATE LEGISLATION’. Available at: https://search-proquest-com.libproxy.ucl.ac.uk/docview/2464567246?pq-origsite=primo.

Howard, J., Huang, A., Li, Z., Tufekci, Z., Zdimal, V., van der Westhuizen, H.-M., von Delft, A., Price, A., Fridman, L., Tang, L.-H., Tang, V., Watson, G. L., Bax, C. E., Shaikh, R., Questier, F., Hernandez, D., Chu, L. F., Ramirez, C. M. and Rimoin, A. W. (2021). ‘An evidence review of face masks against COVID-19’. Proceedings of the National Academy of Sciences, 118 (4), p. e2014564118. doi: 10.1073/pnas.2014564118.

Huang, C., Wang, Y., Li, X., Ren, L., Zhao, J., Hu, Y., Zhang, L., Fan, G., Xu, J., Gu, X., Cheng, Z., Yu, T., Xia, J., Wei, Y., Wu, W., Xie, X., Yin, W., Li, H., Liu, M., Xiao, Y., Gao, H., Guo, L., Xie, J., Wang, G., Jiang, R., Gao, Z., Jin, Q., Wang, J. and Cao, B. (2020). ‘Clinical features of patients infected with 2019 novel coronavirus in Wuhan, China’. The Lancet, 395 (10223), pp. 497–506. doi: 10.1016/S0140-6736(20)30183-5.

Seo, J. and Chung, I.-J. (2019). ‘Face Liveness Detection Using Thermal Face-CNN with External Knowledge’. Symmetry, 11 (3), p. 360. doi: 10.3390/sym11030360.

Sumit Saha. (2018). A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way. Towards Data Science. Available at: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53.

Wang, Y., Hu, M., Zhou, Y., Li, Q., Yao, N., Zhai, G., Zhang, X.-P. and Yang, X. (2020). ‘Unobtrusive and Automatic Classification of Multiple People’s Abnormal Respiratory Patterns in Real Time Using Deep Neural Network and Depth Camera’. IEEE Internet of Things Journal, 7 (9), pp. 8559–8571. doi: 10.1109/JIOT.2020.2991456.

WHO. (2021). ‘Weekly epidemiological update - 2 March 2021’. Who, (March), p. 31. Available at: https://www.who.int/publications/m/item/weekly-epidemiological-update-on-covid-19---23-march-2021.


----

## Declaration of Authorship

I, Xiaohan Feng, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


* Xiaohan Feng *
* 2021/4/29



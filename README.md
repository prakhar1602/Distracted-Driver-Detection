# Distracted Driver Detection 

Prakhar Patidar, Kepan Gao

A majority of severe car accidents are caused by the driver being distracted from the road, with CDC motor vehicle safety division stating, one in five car accidents is caused by a distracted driver [2, 7]. There are 3 major type of distractions defined by CDC as: Visual, Manual and Cognitive where different risks of accident causes are linked to different distraction activities by the driver. Recognizing the distraction activities (with cognitive recognition being out of the scope) via sample images of driver’s dashboard camera could help us possibly alert the driver and also people in nearby vehicles leading to a safer driving experience.

With this project, we aim to take a preliminary step in this direction by focusing on distracted driver detection via sample images of drivers. We will work around building a high-accuracy model capable of classifying whether a driver is driving safely or is distracted doing some other activity. We would be working simultaneously throughout the project life-cycle with no specific defined division of labor as of the moment.
The dataset is sourced from State Farm® [3], containing 22,424 2D dashboard camera images categorized into 10 classes (9 classes defining distraction activities and 1 class labeling each image as safe driving or not). We need to pre-process the images before passing them to our classification algorithms. The color images have to be transformed into high-dimensional matrix based on their RGB values, which would be then resized and flattened to a vector. A class label would be assigned to each such vector based on the category it belongs to prepare the data for classification models.

There has been wide variety of work done in the field of Real-time image based driver distrac- tion detection in computer vision field, with researchers focusing on image preprocessing tech- niques (feature extraction) and classification model selection. One of the main approach had been the use of CNN-based models which have shown to report high accuracy on such problems [1] and generally applied for visual imagery. Another major approach has been the use linear and non-linear Support Vector Machines(SVM) which have also shown promising results in accurately classifying images, although not as accurately as CNN but with faster learning process and lower computational cost than a CNN [5].

Through the course of this project, we would like to explore various classification models such as Support Vector Machine (SVM), Decision Tree, Softmax Classifier, Two-Layer Neural Net- work. Meanwhile, we would also like to consider the pre-trained methods like the Very Deep Convolutional Networks for Large-Scale Image Recognition(VGG) for the better image classi- fication performance [9]. We would like to additionally try implementing Convolution Neural Network (CNN) and ResNet [4], but this would be subjected to vary as per the time constraint. We expect to work with some existing base implementation of these models from Scikit-learn [8] and Pytorch [6] libraries, with additional fine tuning of parameters as required.
1

The ideal outcome of this project, is a successful classification algorithm which is able to detect and classify distracted drivers from the provided input images. We expect to show a comparison between deep learning and decision boundary based classification algorithms observing a compar- ison of validation and test accuracy, precision, recall and computational time for all the models being implemented. The profitability of the system could be evaluated based on the effective de- tection of distracted drivers hence possibly helping us to decrease the risk of accidents.


References


[1] G. E. Hinton A. Krizhevsky, I. Sutskever. Imagenet classification with deep convolutional neural networks. page 2012. https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper. pdf.

[2] CDC. Transportation safety: Distracted driving. https://www.cdc.gov/transportationsafety/distracted_driving/index.html.

[3] State Farm. State farm distracted driver detection. https://www.kaggle.com/c/state-farm-distracted-driver-detection/data.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.

[5] M. A. Hearst, S. T. Dumais, E. Osuna, J. Platt, and B. Scholkopf. Support vector machines. IEEE Intelligent Systems and their Applications, 13(4):18–28, 1998.

[6] Pytorch. Open source machine learning library based on the torch library. https://pytorch.org/.

[7] J.S. Hickman J. Bocanegra R.L. Olson, R.J. Hanowski. Driver distraction in commercial vehicle operations. U.S. Department of Transportation, Washington, DC. https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/docs/ FMCSA-RRR-09-042.pdf.

[8] scikit learn. Free software machine learning library for the python. https://scikit-learn.org/stable/.

[9] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015.
2

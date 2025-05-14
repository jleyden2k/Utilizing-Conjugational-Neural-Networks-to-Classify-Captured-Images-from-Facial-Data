# Utilizing-Conjugational-Neural-Networks-to-Classify-Captured-Images-from-Facial-Data

## Layout

This project was run in Python, with multiple libraries used, including PyTorch, Scikit-Learn, Matplotlib, and Seaborn.

The dataset for Part 1 of this project was obtained from Kaggle, at https://www.kaggle.com/datasets/jonathanoheix/faceexpression-recognition-dataset. The dataset for Part 2 was created manually of facial images from various peers.

## Introduction

Facial expression recognition is a crucial aspect of
human-computer interaction and emotion analysis. Similarly,
hand presence and positioning relative to the face plays a
significant role in facial analysis and recognition. This study
attempts to address these aspects and improve them using similar
approaches. The first part focuses on the creation of a
conjugational neural network (CNN) designed to categorize
human faces into one of seven distinct classes: angry, disgust, fear,
happy, neutral, sad, and surprise. This part utilizes an external
dataset. The second part of this study is focused on training a
similar convolutional neural network. However, this CNN will
categorize images into three distinct classes: no hands, hands
touching face, and hands but not touching face. Unlike in Part 1, a
custom labeled dataset of facial images was used to train the
model, employing similar techniques such as class balancing,
cross-entropy loss, and learning rate scheduling based in ADAM
to optimize performance. The models were trained and validated
for 50 epochs over 5 K-folds, achieving a peak validation accuracy
of 97.99% for Part 1 and 99.40% for Part 2. These results indicate
a highly effective image-based classification model, demonstrating
strong potential for applications in emotional recognition, skin-toskin contact identification, and overall hand-face interaction.
Future work may focus on expanding the dataset to assess model
robustness across the presence of other structures, such as clothes,
masks, or other bodily parts.

## Methods

### Dataset Creation
For Part 1, the dataset was acquired from kaggle.com
(https://www.kaggle.com/datasets/jonathanoheix/faceexpression-recognition-dataset). Fortunately, the dataset seemed
to experience no issues with quality.
For Part 2, Each student was given the option of using
TeachableMachine (https://teachablemachine.withgoogle.com/)
to capture a minimum of 300 images for each class, of which
there were three: Hands Touching Face (Class 1), No Hands
(Class 2), and Hands but Not Touching Face (Class 3). In the
class of 21 students, the participants’ data provided a collection
of approximately 30,000 images, which was encased within a
.zip file and made available for students. For Part 2, the process of creating this custom dataset was
plagued with issues, mainly due to student mistakes (whose
names will remain anonymous). One student mislabeled his/her
image directories, and switched the labels of his/her Class 1 and
Class 2 collections, drastically throwing off validation accuracy.
A different student provided exceptionally low-quality results,
with only about 100 of his/her Class 1 images containing hands
contacting the face. The remainder of these images were the
hands transitioning between different spots on the face or
moving into/out of the frame. Even further, multiple students did
not move their face or hands at all across their images,
essentially resulting in one image being duplicated three to
eleven hundred times. Altogether, these difficulties caused
damage to the quality of the model’s accuracy for all students
using the dataset. These issues were unfortunately not identified
immediately. The model was trained and evaluated over 10
times before the problems with the dataset were identified, in
which the resulting validation accuracies ranged between 7% at
the minimum and 30% at the maximum.

In order to adjust to the problems present in the Part 2
dataset, a manual review of each image was necessary. Each
image was physically reviewed to determine its viability for the
dataset. While this process was needed, it was not strict. Any
image that vaguely resembled the correct category was kept,
while the offending images (and in one case, directory) was
removed. The process was then repeated for the validation set.
As a result, the final dataset contained approximately 15,000
images in the training set and approximately 2,200 images in the
validation set. Immediately after remediation of these dataset
issues, the model was run and evaluated, with the results shown
within this paper. Instant improvements were seen, with the first
attempt beginning at 43% validation accuracy, already 13%
higher than the previous model’s peak performance.

### CNN Model Structure
#### Tech Specifications and Languages
The models design described herein were trained, evaluated,
and run on single Acer Predator PHN-1671 laptop with a
NVIDIA GeForce RTX 4060 GPU. The IDE was Visual Studio
Code, using a Jupyter Notebook run with Python 2.6.0. Due to
the memory and speed constraints of the laptop, it was necessary
to download the latest compatible version of CUDA for GPU
assistance in running the code. Even after this implementation,
the optimized model for Part 1 takes approximately 3.5 hours to
train, even with early stopping at a moderate patience, while Part
2 took approximately 6.5 hours. A large portion of the models
were written through Python libraries, specifically pytorch (plus
torchvision), sklearn, and matplotlib. The usage of PyTorch’s
neural network methods was crucial to the construction of the
CNN, as the models are SimpleCNNs from nn.Module.
#### Data Augmentation
After the initial flaws in the Part 2 dataset were removed,
the number of remaining images was slightly less than desired.
In order to combat the relatively low frequency of data across
both datasets, multiple transformations were performed to
augment it and improve the quality of results. Specifically,
RandomHorizantalFlip, RandomRotation, RandomAffine,
ColorJitter, RandomGrayscale, RandomPerspective, and
RandomErasing were utilized to transform the training data.
Most of these were incorporated at a loss percentage (usually
p=0.1). However, Normalize, batch normalization, and dropout
were also used, with Normalize will all values at 0.5, batch
applied after every convolution, and Dropout at a frequency of
30% (p=0.3). These methods helped tremendously in the
training of the model by improving generalization even further.
Methods for both models in terms of augmentation were
identical.
#### Layer Generation and Normalization
The model utilized in both parts is a deep Convolutional
Neural Network (CNN) designed for classifying images into
categories. While it originally consisted of two convolutional
layers, the final model for Part 1 evolved to include seven
convolutional layers. Due to the addition of hands in Part 2, the
second model was given one additional convolutional layer to
aid in feature extraction. Each convolutional layer was then
followed by batch normalization, max pooling, and a ReLU
activation function. This helped to reduce the spatial
dimensions of the image, while maintaining generalizability.
The convolutional layers initially began at an input size of 3,
from the RBG values of a pixel, and output 16. All batch
values were 32. From there, each layer’s input/output increased
by a factor of 2, to the final layer output of 1042 for Part 1 and
2048 for Part 2. This was flattened, then passed into two
distinct fully-connected (FC) layers. The final layer produced
three output dimensions (one for each class) in softmax format,
as is normal for cross entropy loss functions. A dropout layer
was also added at the end to help prevent overfitting, as
mentioned above.
#### Model Optimization
In both models, the Adam optimizer with weight decay was
utilized for better generalization. The ideal weight decay was
finalized at 1E-5, with an initial learning rate of 1E-4. Further,
cross entropy with class weights was used to address any class
imbalances in the dataset. Additionally, a StepLR learning rate
scheduler was applied to reduce the learning rate over time, at a
step size of 5 and a gamma of 5. Over time, this would
decrease the learning rate, which would help the model to more
accurately find the local minimum. The model was trained
using 5 K-fold stratified cross-validation to ensure robust
performance across different subsets of the dataset,
implemented with scikit learn. To improve training efficiency,
mixed precision training (torch.amp) is enabled, allowing for
faster computation while maintaining numerical stability

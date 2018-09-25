# Self-Driving Car

## Behavioral Cloning

In this part, we used a car simulator to train a Convolutional Neural Network (CNN) to drive a car around tracks of Trend Formula. The Network only eats images from a front facing camera and predict the steering angle, throttle and brake to control the car.

In order to enhance training performance, we use transfer learning methods to use a pre-trained network (VGG16) for base image features and fine-tune a Fully-Connected Neural Network place on top of it.

We could be able to constantly drive around the first track with no issues at all. We are also able to drive through the other tracks through online behavior cloning to fine-tune our model.

## Code usage

```
(keras) [behavioral-cloning]$ tree . -L 2
.
|-- README.md
|-- data
|   `-- Log
|-- drive.py
|-- model.h5
|-- model.json
|-- train.py
`-- tweak.py
```

In this behavior-cloning folder, you will find 3 python scripts to train (train.py), fine-tune (tweak.py) and test (drive.py) the models.

Python Scripts:

* train.py: this file allows you to train a model base on the driving data collected from Trend Formula Simulator.
* tweak.py: this file allows you to fine-tune the pre-trained model by connecting a pygame hook and listening to keyboard input. You will select the 'Autonomous' mode in the Trend Formula Simulator and will correct or tweak the agents driving while it is predicting it's steering angles, throttle and brake around the track. This new data will be collected in memory and the script will train over this data for 10 epochs set once a Ctrl-C signal is sent to it.
* drive.py: this file allows you to let the agent drive autonomously around the selected track.

Model files:

* model.json: represents the model architecture of the Convolutional Neural Network.
* model.h5: represents the trained weights of the model.

## Model Architecture

### Architecture

We decided to use the VGG16 base layers to begin training this network. In specific, the VGG16 network originally contains 16 trainable layers. However, in our case, we only kept the bottom 13 convolutional layers and the corresponding max pooling layers. Additionally, we attached a Fully-Connected Neural Network; this network contained 5 total layers in an attempt to built a Fully-Connected architecture.

`512 -> elu -> dropout -> 256 -> elu  -> dropout -> 64 -> elu  -> dropout -> 3`

### Regularization

Being such a large network and having only 2 tracks to be able to train on, we had to aggresively add regularization methods to our network. First, the VGG16 network come with max pooling layers which progressively reduces the size of the representation in order to reduce the amount of parameters and computation in the network, this way helps in the prevention of overfitting. The base layers of the VGG16 network contains 5 blocks of 2-3 convolutions each. Each of these blocks has a max pooling layer at the end of its last convolution.

Additionally to the max pooling layers, we added a dropout layer after each of the dense layers on the top network. These dropout layers also work to reduce overfitting by randomly dropping a specified number of connections each forward pass, therefore forcing the network to learn important features in different neurons, and simultaneously preventing complex adaptations to the training data.

Finally, we use an early stopping function that would be queried after each epoch. The function will basically stop training if it found that the validation loss was improving for the past 2 epochs.

## Training Strategy

### Feature Transfer

For our training, we first leverage the feature layers of the VGG16 network. The network we started with had it's weights previously trained on the 'ImageNet' data set.

First, we set all of the pre-trained VGG16 features layer so that they are not trainable. In other words, it's weight will not change. Then, we connect the top layer and train a single epoch of a subset of the data. This allows for the weights of the top layers to be initialized to sensible values that work with the features currently on the VGG layers.

Next, we proceed to make only the upper 2 blocks of the VGG16 network trainable. This is so that the base features learned with the 'ImageNet' training stay in place. These features are only the base and therefore will consist of simple shapes and line like circles, squares and so on. Enabling these upper 2 blocks for training is very important because we want the network to 'forget' about cats, planes and horses, and learn about roads, trees, mountain, lane-lines, etc. These new items will most likely be learned on the top 2 layers only and the whole network is therefore able to learn new things.

Finally, we train on the whole data for several epochs until the loss in the validation set stops improving.

### Data Collection

Collecting the data was one of the most important steps during this global AI contest. In fact, one of the most important things we learned is that "is all about the data". This couldn't have been more true.

One of the main issues we had to collect the data is that the keyboard doesn't do justice for the way steering wheels work. For example, if we pressed the left key, the simulator would receive a left signal and the magnitude of this signal would increase exponentially. So, the longer you pressed the key, the more the turning would add up to be. In real life, on is able to keep a constant value on the steering wheel when turning which makes for a much easier training environment. Additonally, when we release the key, the signal goes immediatelly to a flat 0. This is obviously, not only challenging, but just not the way cars work in real life.

In addition, after using this driving strategy and training the base
 model, we developed a tweak script that would intercept the trained agent's 
predictions and merge them with a user input. This script is very useful for 
specific corrections on a working model. For example,
the script uses the same main code as in train.py, but it also listens to key 
presses from the user. If the left or right keys are pressed, a very small 
value would be added to the predicted value so as to 'fix' an already good 
prediction. These image and adjusted prediction values would be stored on a
 numpy array and use for later for training. When the user feels like got 
enough and useful corrections on the current driving behavior, a simple 
Ctrl-C will kill the server sending the messages to the simulator and 
engage in training. The data is passed through with a very small learning 
rate so as to not damage or overfit the data, and for only a few epochs.

### Hyper-parameter Tunning

Since we used an Adam optimizer, most of the hyper parameter tuning is 
done internally on it. However, we did select a learning rate that is high 
enough so to train fast, and not so high so that the agent would actually
 learn. Additionally, we use a decay rate of about 0.7, this way the 
initial learning rate will decrease and the model will get more precise 
as the epochs increased.

### Batch Generator

We developed a batch generator that would allow us to yield unlimited combinations
of images in batches as required. In this batch generation function we added a couple
of special features. First, the image pre-processing function would be called on
on the images to be added to the batch. This reduced the amount of computation
and additionally improve training time because these data augmentation
strategies would run on the CPU while the training process would run on the GPU.
The pre-processing of the image included image resize to a 100x200 ratio and then
cropping the image to 40 rows below the horizon as shown in the image.

Then, we do image normalization with values laying in between -1 and 1.
We tried using image convertion to YUV space instead of RGB,
but perhaps since we used transfer learning and the 'ImageNet'
weights had been acquired with RGB images, it
was better not to do so.

After the image was resized and normalized, it was horizontally flipped
with a 50% chance. If the image was flipped, then the corresponding labels
was multiplied by -1 to look for the corresponding turning angle. Then the images
would be appended to a batch of 128 and passed to the training procedure.


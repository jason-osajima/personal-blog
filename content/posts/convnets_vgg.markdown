title: Convolutional Networks - VGG16 
slug: convnets_vgg
category: 
date: 2018-08-18
modified: 2018-08-18
tags: neural networks, machine learning, convolutional networks, VGG

## Introduction
<!-- PELICAN_BEGIN_SUMMARY -->
The Imagenet Large Scale Visual Recognition Challenge ([ILSVRC](http://www.image-net.org/challenges/LSVRC/)) is an annual computer vision competition. Each year, teams compete on two tasks. The first is to detect objects within an image coming from 200 classes, which is called object localization. The second is to classify images, each labeled with one of 1000 categories, which is called image classification.

In 2012, Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton won the competition by a sizable margin using a convolutional network (ConvNet) named [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). This became a watershed moment for deep learning.

Two years later, Karen Simonyan and Andrew Zisserman won 1st and 2nd place in the two tasks described above. Their model was also a ConvNet named VGG-19. VGG is the acronym for their lab at Oxford (Visual Geometry Group) and 19 is the number of layers in the model with trainable parameters.

What attracted me to this model was its simplicity - the model shares most of the same basic architecture and algorithms as [LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), one of the first ConvNets from the 90s. The main difference is the addition of several more layers (from 5 to 19), which seems to validate the idea that deeper networks are able to learn better representations (this trend continues with the introduction of Residual Networks, which won IVCLR the following year with a whopping 152 layers).

<!-- PELICAN_END_SUMMARY -->

Similar to my first post on [forward propagation](/forwardprop) and [backpropagation](/backprop) for a vanilla neural network, I will walk through forward propagation and backpropagation for VGG-16 and discuss some of the advantages of using a ConvNet over a fully-connected neural network for computer vision tasks. VGG-16 is comparable in performance to VGG-19 but is simpler (it has three fewer layers) so we will roll with that.

## Objective

The objective for this task is to predict 5 (out of 1000) classes for each one of 100,000 test set images. The actual (ground truth) image label has to be one of the five predicted classes.

So our input will be the pixel values of an image. The images in the ILSVRC dataset are fixed-size 224 x 224 RGB. What this means is that for each color channel (red, green and blue) each image has 224 x 224 pixel values. We can represent each input as a tensor with dimensions $(224, 224, 3)$ and label it as $\textbf{x}$. 

Each pixel value is a scalar and can take on values between 0 and 225. We can represent a pixel value as $x_{(i,j,k)}$, where i is the value for the first dimension, j for the second, and k for the third (which recall is the channel).

The first two dimensions represent the location of the pixel value within the image, and the third dimension is the channel that pixel value belongs to (so for example, if the third dimension of a pixel value is equal to 1, it belongs to the red channel).

Representing the image as a tensor looks something like this:

{% img /images/input_x.png 400 [input_x] %}

Our output $\hat{\textbf{y}}$ will be a vector of probabilities for each of the 1000 classes for the given image. 

$$
\hat{\textbf{y}} =
\begin{bmatrix}
\hat{y}_{1} \\\\
\hat{y}_{2} \\\\ 
\vdots \\\\
\hat{y}_{1000}
\end{bmatrix}
$$

Let's say we input an image $\textbf{x}$ and the model believes the image belongs to class 1 with probability 5%, class 2 with probability 7%, class 4 with probability 9%, class 999 with probability 2%, class 1000 with probability 77%, and all other classes with probability 0%. In this situation, our output $\hat{\textbf{y}}$ would look like this:

$$
\hat{\textbf{y}} =
\begin{bmatrix}
0.05 \\\\ 
0.07 \\\\
0 \\\\
0.09 \\\\ 
\vdots \\\\
0.02 \\\\
0.77
\end{bmatrix}
$$

Notice that the sum of the elements in $\hat{\textbf{y}}$ are equal to $1$. We use a softmax function at the end of our ConvNet to ensure this property. We'll discuss the softmax function later.

In order to compute accuracy, we use $\hat{\textbf{y}}$ to create a vector of the top 5 classes in decreasing order of probability. 

$$
\textbf{c} =
\begin{bmatrix}
c_1 \\\\ 
c_2 \\\\
c_3 \\\\
c_4 \\\\ 
c_5
\end{bmatrix}
$$

Using our example, we get:

$$
\textbf{c} =
\begin{bmatrix}
1000 \\\\ 
2 \\\\
1 \\\\
4 \\\\ 
999
\end{bmatrix}
$$

Let's say the image was an image of a dog on a street. The picture of the dog is the 4th class. The picture of the street is the 1000th class. Since the image has two ground truth classes, we get:

$$
\textbf{C} =
\begin{bmatrix}
C_1 \\\\ 
C_2
\end{bmatrix}
$$

Using our example, we get for our ground truth labels:

$$
\textbf{C} =
\begin{bmatrix}
4 \\\\ 
1000
\end{bmatrix}
$$

Let $d(c_i, C_k) = 0$ if $c_i = C_k$ and 1 otherwise. To calculate the error of the algorithm, we use:

$$e = \dfrac{1}{n} \sum_k \min_i d(c_i, C_k)$$

For our example, the error is:

$$e = \dfrac{1}{2}\big( \min_i d(c_i, C_1) + \min_i d(c_i, C_2) \big)$$
$$e = \dfrac{1}{2}\big( 0 + 0 \big)$$
$$e = 0$$

Since both of the ground truth labels were in our top 5, we get an error of $0$.

### Considering Batches of $m$ training examples

We just deconstructed the input and output for our model. Hopefully it makes sense. So far, we've only considered the input and output for one example. What if we had $m$ training examples that we wanted to input into the model as a batch? It's actually not too bad, we are just going to add another dimension.

$$
\textbf{X} = [\textbf{x}^{(1)}, \textbf{x}^{(2)}, ... \textbf{x}^{(m)}]
$$

So we can think of $X$ as a tensor with dimensions $(m, 224, 224, 3)$, where $m$ is the number of examples in our batch. The superscript denotes which training example it is in the batch. So $\textbf{x}^{(1)}$ would be the 1st training example from the batch. Similarly for output:

$$
\textbf{Y} = [\textbf{y}^{(1)}, \textbf{y}^{(2)}, ... \textbf{y}^{(m)}]
$$

$Y$ is a tensor with dimensions $(1000, m)$, and we use the same superscript notation as above.

## Defining the Architecture

Let's see if we can represent all 16 layers of this model visually:

{% img /images/VGG_1.png [VGG_1] %}

*Diagram of the architecture of VGG-16*

If you notice, layers are represented as either 3D rectangular prisms (the layers on the left) or 2D rectangles (the layers on the right). This was done on purpose to represent the dimensions of the layer. For example, recall that the input $\textbf{x}$ is a 3D tensor $(224, 224, 3)$ and is represented on the far left as a prism. Our output $\hat{\textbf{y}}$ is a matrix $(1000, 1)$ and is represented on the far right as a rectangle.

Alright. If you are crazy you might have counted the layers and noticed that there are 24 layers in this diagram. But this model is called VGG-16. So the 16 refers to the number of layers that have trainable parameters. I'll highlight and label them - 

{% img /images/VGG_2.png [VGG_2] %}

*Diagram of the architecture of VGG-16 with trainable parameters highlighted in red.*

There are two types of layers with trainable parameters that are highlighted. The 3D ones on the left are Conv Layers, and the ones on the right are Fully-connected Layers.

What are the other layers that don't have trainable parameters? Recall that the layers on the far left and far right are the input $\textbf{x}$ and output $\hat{\textbf{y}}$ layers. In order to get the output $\hat{\textbf{y}}$ layer we apply the Softmax function, so we will call this layer the Softmax Layer.

{% img /images/VGG_3.png [VGG_3] %}

*Diagram of the architecture of VGG-16 with input and output highlighted in blue*


The layers that follow the string of Conv Layers are called Pooling Layers.

{% img /images/VGG_4.png [VGG_4] %}

*Diagram of the architecture of VGG-16 with Pooling Layers highlighted in green*


Finally, there's a layer where we flatten the 3D tensor into a column vector. I don't think this layer has an official name, so we'll call it a Flat Layer.

{% img /images/VGG_5.png [VGG_5] %}

*Diagram of the architecture of VGG-16 with Flat Layer highlighted in purple.*

So within this architecture, there are:

* Conv Layers
* Pooling Layers
* Flat Layer
* Softmax Layer

In the next few sections, we will take an example from each and describe how the math works. We will ask the question, "what operations are we applying to the previous layer to get to the current layer?" We will start with Conv Layers, which are the integral part of Conv Nets.

## Conv Layers

In [The Deep Learning Book](https://www.deeplearningbook.org/contents/convnets.html), the authors describe the difference between Conv Nets and Neural Networks:

>Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

Recall in our previous [neural network example](/forwardprop), the transition from the first layer $\textbf{a}^{[0]}$ to the second layer $\textbf{a}^{[1]}$ was simply a weight matrix multiplication, the addition of a bias, and the element-wise application of the ReLU function.

$$\textbf{a}^{[1]} = g(\textbf{W}^{[1]}\textbf{a}^{[0]} + \textbf{b}^{[1]})$$

For VGG-16, the only difference that we will be making is replacing the general matrix multiplication with a convolution, and instead of a 2-D weight matrix, we will be using a 3-D filter tensor.

Let's deconstruct the first Conv Layer in the diagram, the one that succeeds the input layer ($\textbf{x}$). 

{% img /images/VGG_6.png [VGG_6] %}

*Diagram of the architecture of VGG-16 with example Conv Layer and preceding input layer highlighted.*

Let's first cover the dimensions of each. We discussed earlier that our input is a picture, with dimensions 224 by 224 by 3. We labeled this tensor $\textbf{x}$:

{% img /images/input_x.png 400 [input_x] %}

As in our previous post on forward prop, we can think of this layer as $\textbf{a}^{[0]}$. Let's call the first Conv Layer $\textbf{a}^{[1]}$. This layer has dimensions 224 by 224 by 64:

{% img /images/VGG_7.png 400 [VGG_7]  %}

*Diagram of first Conv Layer $\textbf{a}^{[1]}$ with dimensions $(224, 224, 64)$*

So how do we go from $\textbf{x}$ to $\textbf{a}^{[1]}$? We start by applying the convolution function to $\textbf{x}$ 6 times using 6 different filters and then add a bias $\textbf{b}^{[1]}$ to get our intermediate product, $\textbf{z}^{[1]}$. We'll call the collection of filters $\textbf{W}^{[1]}_c$, with dimensions $(16, 16, 3, 6)$. The first three dimensions represent the height, width, and number of channels, and the last dimension is the filter. We can represent each filter with $\textbf{W}^{[1]}_c(i)$, where $i$ is the filter number in $\textbf{W}^{[1]}_c$. 

We add the same bias for each filter, so $\textbf{b}^{[1]}$ has dimensions $(6,1)$.

Let's see if we can represent this visually:

{% img /images/VGG_8.png 600 [VGG_8]  %}

*Diagram of transition from $\textbf{x}$ to $\textbf{z}^{[1]}$. A convolution is applied to $\textbf{x}$ 6 times using 6 different filters $\textbf{W}^{[1]}_{c(1)}$, $\textbf{W}^{[1]}_{c(2)}$, ... $\textbf{W}^{[1]}_{c(6)}$*

Let's simplify the dimensions to make it easier to visualize. $\textbf{x}$ is a tensor with dimensions $(224, 224, 3)$. Let's instead make it $(5, 5, 3)$. $\textbf{W}^{[1]}_{c}$ has dimensions $(16, 16, 3, 64)$. Let's make it $(3,3,3,6)$. We won't change the dimensions for $\textbf{b}^{[1]}$. Finally, for $\textbf{z}^{[1]}_{(1)}$ we'll change it's dimensions from $(224, 224, 64)$ to $(3,3,6)$.

{% img /images/VGG_9.png [VGG_9]  %}

*Diagram of convolution between the simplified $\textbf{x}$ and the 6 filters of $\textbf{W}^{[1]}_{c}$, resulting in the 6 channels of $\textbf{z}^{[1]}$.*

### Casting the Bias ($\textbf{b}^{[1]}$)

If you noticed in our diagram, for each of the six elements in $\textbf{b}^{[1]}$, we repeated it several times to create a tensor that matched the shape of $\textbf{z}^{[1]}_(i)$ so for example, the scalar $b^{[1]}_{(1,1)}$ was converted into a tensor of shape $(3,3,3)$. We do this because the addition of the bias is elementwise, meaning we add the bias to each element of the product of our convlution.

So how do we get a value for $z^{[1]}_{(i,j,k)}$ for a given row $i$, column $j$, and channel $k$ in $z^{[1]}$? Unsurprisingly, we use the convolution function:

$$z^{[1]}_{(i,j,k)} = (\textbf{x} * \textbf{W}^{[1]}_{c})(i,j,k) + b^{[1]}_{(k,1)}$$

Which becomes:

$$z^{[1]}_{(i,j,k)} = \sum^{3}_{l = 1}\sum^{3}_{m = 1}\sum^{3}_{n = 1}x_{(i + l - 1, j + m - 1, n)}W^{[1]}_{c(l,m,n,k)} + b^{[1]}_{(k,1)}$$

>Technically, this is not a convolution but a related function called the cross-correlation. But most deep learning libraries and papers use cross-correlation as the convolution function.

So we have defined $i$, $j$, $k$ as the coordinates of our end product, $z^{[1]}$, but what does $l$, $m$, and $n$ represent? If you look at the equation above, $l$ is the row number of the filter, $m$ is the column number of the filter, $n$ is the channel number and $k$ tells you which filter we are using. Notice that the convolution is similar to the matrix multiplication we did with a fully-connect neural network, since we are just multiplying elements of $\textbf{x}$ and $\textbf{W}^{[1]}$. We'll talk about the differences a little later, but for now just relish in the fact that the actual math is just multiplication.

Let's figure out how we get the value for $z^{[1]}_{(1,1,1)}$. Plugging in 1 for $i$, $j$, and $k$, we get:

$$z^{[1]}_{(1,1,1)} = (\textbf{x} * \textbf{W}^{[1]}_{c})(1,1,1)+ b^{[1]}_{(1,1)}$$
$$z^{[1]}_{(1,1,1)} = \sum^{3}_{n = 1}\sum^{3}_{l = 1}\sum^{3}_{m = 1}x_{(1 + l, 1 + m, n)}W^{[1]}_{c(l,m,n,1)} + b^{[1]}_{(1,1)}$$

If you think about it, we are summing over the three dimensions (row, column, channel) of the 1st filter. We know to use the first filter because $z^{[1]}_{(1,1,1)}$ has 1 as it's final dimension. Let's sum over the row and column first and see what we come up with:

$$z^{[1]}_{(1,1,1)} =
\sum^{3}_{n = 1} \bigg(
x_{(1, 1, n)}W^{[1]}_{c(1,1,n,1)} + 
x_{(2, 1, n)}W^{[1]}_{c(2,1,n,1)} + 
x_{(3, 1, n)}W^{[1]}_{c(3,1,n,1)} +
$$
$$ 
x_{(1, 2, n)}W^{[1]}_{c(1,2,n,1)} + 
x_{(2, 2, n)}W^{[1]}_{c(2,2,n,1)} + 
x_{(3, 2, n)}W^{[1]}_{c(3,2,n,1)} +
$$
$$ 
x_{(1, 3, n)}W^{[1]}_{c(1,3,n,1)} + 
x_{(2, 3, n)}W^{[1]}_{c(2,3,n,1)} + 
x_{(3, 3, n)}W^{[1]}_{c(3,3,n,1)} 
\bigg) + b^{[1]}_{(1,1)}$$

So recall that each channel of our first filter has dimensions $(3,3)$ which means that it has 9 values total. So it would make sense that we would have 9 terms in the above equation. Visually, it looks like this:

{% img /images/VGG_10.png 400 [VGG_10]  %}

*Diagram of operations needed to calculate $z^{[1]}_{(1,1,1)}$. Notice that for each channel of $\textbf{x}$, there is a corresponding channel in the first filter of $\textbf{W}^{[1]}_{c}$. We multiply each value of $\textbf{x}$ that lines up with $\textbf{W}^{[1]}_{c}$, and then add them all together. So we have a total of 18 values that we are adding together.*

If we finally sum over the channels, we get:

$$z^{[1]}_{(1,1,1)} =
x_{(1, 1, 1)}W^{[1]}_{c(1,1,1,1)} + 
x_{(2, 1, 1)}W^{[1]}_{c(2,1,1,1)} + 
x_{(3, 1, 1)}W^{[1]}_{c(3,1,1,1)} +
$$
$$ 
x_{(1, 2, 1)}W^{[1]}_{c(1,2,1,1)} + 
x_{(2, 2, 1)}W^{[1]}_{c(2,2,1,1)} + 
x_{(3, 2, 1)}W^{[1]}_{c(3,2,1,1)} +
$$
$$ 
x_{(1, 3, 1)}W^{[1]}_{c(1,3,1,1)} + 
x_{(2, 3, 1)}W^{[1]}_{c(2,3,1,1)} + 
x_{(3, 3, 1)}W^{[1]}_{c(3,3,1,1)} +
$$
$$
x_{(1, 1, 2)}W^{[1]}_{c(1,1,2,1)} + 
x_{(2, 1, 2)}W^{[1]}_{c(2,1,2,1)} + 
x_{(3, 1, 2)}W^{[1]}_{c(3,1,2,1)} +
$$
$$ 
x_{(1, 2, 2)}W^{[1]}_{c(1,2,2,1)} + 
x_{(2, 2, 2)}W^{[1]}_{c(2,2,2,1)} + 
x_{(3, 2, 2)}W^{[1]}_{c(3,2,2,1)} +
$$
$$ 
x_{(1, 3, 2)}W^{[1]}_{c(1,3,2,1)} + 
x_{(2, 3, 2)}W^{[1]}_{c(2,3,2,1)} + 
x_{(3, 3, 2)}W^{[1]}_{c(3,3,2,1)} +
$$
$$
x_{(1, 1, 3)}W^{[1]}_{c(1,1,3,1)} + 
x_{(2, 1, 3)}W^{[1]}_{c(2,1,3,1)} + 
x_{(3, 1, 3)}W^{[1]}_{c(3,1,3,1)} +
$$
$$ 
x_{(1, 2, 3)}W^{[1]}_{c(1,2,3,1)} + 
x_{(2, 2, 3)}W^{[1]}_{c(2,2,3,1)} + 
x_{(3, 2, 3)}W^{[1]}_{c(3,2,3,1)} +
$$
$$ 
x_{(1, 3, 3)}W^{[1]}_{c(1,3,3,1)} + 
x_{(2, 3, 3)}W^{[1]}_{c(2,3,3,1)} + 
x_{(3, 3, 3)}W^{[1]}_{c(3,3,3,1)} + b^{[1]}_{(1,1)}
$$

Ok great! So we figured out how to calculate $z^{[1]}_{(1,1,1)}$. Only $4 * 4 * 6 - 1 = 95$ more calculations to go. Just kidding, we aren't going to go through each calculation. I do want to talk about how we decide which values of $\textbf{x}$ we choose to multiply with the values in $W^{[1]}_{c}$. If we follow the formula that we outlined earlier, we end up 'sliding' over the values of $\textbf{x}$. I think a gif will help demonstrate what I mean:

{% img /images/conv.gif 400 [conv_gif]  %}

So in this gif, we calculated $z^{[1]}_{(1,1,1)}$ through $z^{[1]}_{(3,3,1)}$, which are all the outputs that depend on the first filter of $W^{[1]}_c$. We would then continue this process 5 more times for the other 5 filters of $W^{[1]}_c$. The final dimensions of our output, $z^{[1]}$ would therefore be $(3,3,6)$.

Notice that each channel in our output shrunk in height and width by $2$, from $(5,5)$ to $(3,3)$. Is this a problem? Yes, because it has been demonstrated that deeper architectures (more layers) perform better. If we are shrinking our dimensions from layer to layer, we are losing a lot of information. We could reduce the size of conv filters ($W_{c}$), but if we do that we are limiting the ability of the filter to learn representations within the data.

The creators of VGG-16 recognized this problem, and used something which allowed them to maintain the height and width when they transitioned between the input and output of a conv layer. That something is called same padding.


### Same Padding

In order to prevent the problem of the height and width of layers shrinking, many people use something called same padding. Same padding allows us to maintain the height and width between the input and output of a convolution. Let's go back to our simplified example. Currently, we transition from $\textbf{x}$, which is $(5,5,3)$ to $\textbf{z}^{[1]}$, which is $(3,3,3)$. If we wanted $\textbf{z}^{[1]}$ to have $(5,5,3)$ we could 'pad' $\textbf{x}$ with 0's. Let's define this transformation as $\textbf{s}^{[0]} = h_p(\textbf{x})$, where $p$ is equal to the number of borders of $0$ we place around $\textbf{x}$. 

{% img /images/VGG_11.png 300 [VGG_11]  %}

*$\textbf{x}$ padded with one border of $0$'s. We define this new tensor as $\textbf{s}^{[0]}$*

In the image above, $\textbf{x}$ is padded with one border of $0$'s. How do we know how many $0$'s to add to make sure that our input $\textbf{x}$ shares the same height and width as $\textbf{z}^{[1]}$? There's a pretty easy formula to figure that out! So given a filter with height $f$ and width $f$ the padding ($p$) is equal to:

$$p = \dfrac{f-1}{2}$$

By convention in computer vision, $f$ is almost always odd so we don't need to worry about a non-whole number padding.

> A quick note: All of the convolutional layers in VGGNet use the same stride size of 1. So we won't go into how this calculation changes when we increase the stride size.

For our example filter, $f = 3$ and therefore we need $p = 1$ to use same padding.

{% img /images/pad.gif 400 [pad_gif]  %}

Our gif now ticks up to $\textbf{z}^{[1]}_{(5,5,1)}$, which shares the height and width of $\textbf{x}$ and is what we wanted. 

Alright, so now we've learned how to calculate $\textbf{z}^{[1]}$, and it shares the same dimensions as $\textbf{x}$ thanks to same padding. We need to do one more operation in order to get to our first hidden layer, $\textbf{a}^{[1]}$ which is the ReLU.

### ReLU Operation

In my previous post describing [forward propagation](/forwardprop) in a fully connected network, I talked about how a ReLU function works. ReLU is implemented for convolutional networks the same way it's implemented for fully connected networks. If our ReLU function is $g$, then:

$$\textbf{a}^{[1]} = g(\textbf{z}^{[1]})$$

$g$ is applied elementwise to every element in $\textbf{z}^{[1]}$. As we would expect, $\textbf{a}^{[1]}$ is $(3,3,6)$ and shares the same dimensions as $\textbf{z}^{[1]}$.

With that, we have successfully transitions from the input layer $\textbf{x}$ to the first hidden layer, $\textbf{a}^{[1]}$. Before moving on I think it would be useful to talk about why we use convolutional layers instead of fully connected layers. Most of the following conversation comes from the chapter on Convolutional Networks from the [Deep Learning Book](http://www.deeplearningbook.org/contents/convnets.html) and I highly recommend you check that out.

## Why ConvNets?

There are two main reasons why we use convolutional layers over fully connected layers: **sparse interactions** and **parameter sharing**.

### Sparse Interactions

What if we had used a fully connected layer instead of a convolutional layer in the preceding example? Recall that our input $\textbf{x}$ had $224 * 224 * 3 = 150,528$ elements. The first hidden layer $\textbf{a}^{[1]}$ had $224 * 224 * 6 = 301,056$ elements. If we connected these two layers fully, we would need a weight for each combination. Which means we would need $150,528 * 301,056= 45,317,357,568$ weights, plus $301,056 * 1 = 301,056$ biases. Good lord, and that's just for the first layer.

In contrast, our first convolutional layer has $16 * 16 *  3 *  6 = 4608$ weight parameters and $6 * 1 = 6$ bias parameters. While we share those parameters between inputs (which we will discuss next), those parameters are connected to vastly less inputs than the parameters of a fully connected layer would, and therefore conv layers require orders of magnitude less memory (fewer parameters) and runtime (fewer operations).

### Parameter Sharing

In a fully connected layer, we use each weight parameter one time, since each weight parameter connects one input element to one output element. In contrast, in a convolutional layer, we reuse each weight parameter multiple times. In the example above, the weight parameter  $W^{[1]}_{c(1,1,1,1)}$ is used a total of ($5 * 5 = 25$) times, multiplying it by the input $\textbf{x}$ after padding it with one border of 0's ($p = 0$) to get $\textbf{s}^{[0]}$. The elements of $\textbf{s}^{[0]}$ that we use are displayed below: 

$$
s^{[0]}_{(1,1,1)},
s^{[0]}_{(1,2,1)},
s^{[0]}_{(1,3,1)},
s^{[0]}_{(1,4,1)},
s^{[0]}_{(1,5,1)},
$$
$$
s^{[0]}_{(2,1,1)},
s^{[0]}_{(2,2,1)},
s^{[0]}_{(2,3,1)},
s^{[0]}_{(2,4,1)},
s^{[0]}_{(2,5,1)},
$$
$$
s^{[0]}_{(3,1,1)},
s^{[0]}_{(3,2,1)},
s^{[0]}_{(3,3,1)},
s^{[0]}_{(3,4,1)},
s^{[0]}_{(3,5,1)},
$$
$$
s^{[0]}_{(4,1,1)},
s^{[0]}_{(4,2,1)},
s^{[0]}_{(4,3,1)},
s^{[0]}_{(4,4,1)},
s^{[0]}_{(4,5,1)},
$$
$$
s^{[0]}_{(5,1,1)},
s^{[0]}_{(5,2,1)},
s^{[0]}_{(5,3,1)},
s^{[0]}_{(5,4,1)},
s^{[0]}_{(5,5,1)},
$$

So how does parameter sharing help? Well, let's say that one filter detects the edges in a picture. We only need one set of weights to do this job across the entire image for each channel, since the operation won't change as we move across the image. In this case, parameter sharing is more efficient than using one weight parameter per pixel to connect it to the next layer. 

A special case of parameter sharing is equivariance. CNN's are equivariant in the sense that if we translate an image of a corgi dog across an image, for example, the output will be the same, but just translated to where the corgi is in the image. This is not true for rotations or scale however, and these need to be handled separately.

In this section, we've discussed the transition from our input $\textbf{x}$ to our first conv layer, $\textbf{a}^{[1]}$. In order to make this transition, we transformed $\textbf{x}$ to $\textbf{s}^{[0]}$ using same padding. Next, we applied a convolution using filter weights $\textbf{W}^{[1]}_c$ and added a bias $\textbf{b}^{[1]}$ to get $\textbf{z}^{[1]}$. Next, we used the ReLU activation function to introduce nonlinearity elementwise to get $\textbf{a}^{[1]}$.

$$\textbf{x} \rightarrow \textbf{s}^{[0]} \rightarrow \textbf{z}^{[1]} \rightarrow \textbf{a}^{[1]}$$

Within VGG16, we use this same procedure 13 times, until we flatten our output and use 3 fully connected layers. Between conv layers, the creators of VGG16 interspersed pooling layers, which are used for downsampling. We will discuss them next.


# Pooling Layer

We will focus our attention next on the pooling layer. In particular, we will focus on the transition between a conv layer $\textbf{a}^{[2]})$ and the first pooling layer $\textbf{m}^{[2]}$ in the architecture.

{% img /images/VGG_12.png [VGG_12]  %}

*Diagram of the architecture of VGG-16 with example Pooling Layer $\textbf{m}^{[2]}$ and preceding Conv Layer $\textbf{a}^{[2]}$ highlighted.*

We'll first deconstruct what happens in the pooling layer of the VGG16 architecture and then discuss the motivation behind pooling layers. Let's start! VGG16 uses a particular type of pooling operation called max pooling. The dimensions of the input $\textbf{a}^{[2]}$ are $(224, 224, 64)$ and the dimensions of the output $\textbf{m}^{[2]}$ are $(112, 112, 64)$.

Similar to our last example, let's simplify the dimensions to make it easier to work through the example. Let's make $\textbf{a}^{[2]}$ have dimensions $(6, 6, 6)$ and $\textbf{m}^{[2]}$ have dimensions $(3, 3, 6)$. 

Pooling has some similarities with convolutions. Like the convolutional operation, we are sliding over our input and performing a pooling operation. VGG16 uses max pooling, which takes the max value within the window. If we were to compute the $m^{[2]}_{(i,j,k)}$, it would look like this:

$$m^{[2]}_{(i,j,k)} = \max_{i * s <= l < i * s + f, j * s <= l < j * s + f }a^{[2]}_{(l,m,k)}$$

Similar to our previous example, $i$ is the height, $j$ is the width, and $k$ is the channel of the output $\textbf{m}^{[2]}$. $l$ and $m$ are the height and width of our input. The max pooling filter has height and width equal to $f$ and $s$ is the stride size. Stride size indicates how many elements we pass over for our operation. In our previous example, we used a stride size of 1. In this case, we use a stride size of 2. 

How do we figure out how big of a filter to use to get an output that has dimensions $(3,3,6)$ when our input has dimensions $(6,6,6)$? We can use a pretty useful formula. Assuming our input's height and width are equal and our output's height and width are equal, let's let $n_L$ represent the height and width of our input (in this case, $\textbf{a}^{[2]}$) and $n_{L+1}$ represent the height and width of our output (in this case, $\textbf{m}^{[2]}$). To find the height and width ($f$) of the window we need to use for max pooling, we use:

$$n_{L+1} = \bigg \lfloor \dfrac{n_L + 2p - f}{s} + 1 \bigg \rfloor$$

Solving for $f$:

$$f = n_L + 2p - s(n_{L+1} - 1)$$

Where $p$ is equal to the padding. Since we aren't using padding, we set $p = 0$. And plugging in $n_L = 6$, $s = 2$, and $n_{L+1} = 3$, we get:

$$f = 6 + 2*0 - 2(3 - 1)$$
$$f = 2$$

So the height and width of our window should be equal to $2$. And just to be clear, this is a pooling window, which means that it's not a filter of trainable parameters like in a convolutional layer. But, we can use this same formula to calculate dimensions of a filter in a convolutional layer.

Plugging in $s = 2$, $f = 2$ into our equation, we get:

$$m^{[2]}_{(i,j,k)} = \max_{i * 2 <= l < i * 2 + 2, j * 2 <= l < j * 2 + 2 }a^{[2]}_{(l,m,k)}$$

So essentially, to get an element in $\textbf{m}^{[2]}$, we take a $2x2$ window of $\textbf{a}^{[2]}$ and return the maximum value in that window. We then slide over by $2$ and do it again.

Great, so now that we know how the max pooling operation works, why do we use it?

### Why Max Pooling?

There are a few reasons why we use max pooling. The first is that with each set of convolutional layers in VGG16, you may notice that we are increasing the depth, or the amount of channels. In the layer before we flatten our tensor to use it in a fully-connected layer, we have a depth of $512$. Depth is important because it signfies the structured information that the network has learned about the input. However, it would not be memory efficient to maintain our original height and width of $224$ and end up with a depth of $512$. Pooling allows us to take a summary statistic (in this case, the max) of a window within a convulational layer and send it to the next level. In essence, we are roughly taking the most important 'activations' from the previous layer and sending it to the next layer, thereby reducing the height and width and decreasing the memory requirements. As an example of this, in the final pooling layer, we end up with a tensor with dimensions $(7,7,512)$. This is sometimes called downsampling.

> You can see that this trend happens in many convolutional network architectures. As we go deeper, we increase the depth or number of channels of our layer (as a result of a convolution) and decrease the height and width of our layer (as a result of pooling)

The second is it makes the network invariant to small translations in the input. What this means is that we change the input slightly, the max pooling outputs will stay the same since it will still report the maximum value in the window. This is important in image classification, because the location of, say a nose, won't always be in the same location at all times.

Alright, so we've discussed an example of a pooling layer. Next, we will briefly talk about the flat layer and softmax layers.

## Flat Layer

In a Flat Layer, we take as input the final max pooling layer ($\textbf{m}^{[13]}$) and flatten it, to get as output a flat layer $\textbf{f}^{[13]}$ with dimensions $(25,088, 1)$. The $25,088$ comes from multiplying all the dimensions of the input layer ($7 * 7 * 512 = 25,088$). The reason that we do this is because fully connected layers take as input a row (or column, depending on the math notation) vector as opposed to a tensor. So, in this layer nothing too crazy happens, we are just changing the dimension to prepare to use the fully connected layers.

{% img /images/VGG_13.png [VGG_13]  %}

*Diagram of the architecture of VGG-16 with example Flat Layer $\textbf{f}^{[13]}$ and preceding Pooling Layer $\textbf{m}^{[13]}$ highlighted.*

## Fully Connected Layer

After our 13 convolutional layers, we connect our flat layer to 3 fully connected layers. In a [previous post](/forwardprop) I talk about how fully connected layers work so I won't go into too much detail about them here. What I do want to discuss is why we use fully connected layers at all. The reason why we wouldn't want to use them is the huge amount of weight parameters in the first fully connected layer:

>$512 * 7 * 7 * 4096 = 102,760,448$ weight parameters connected the flat layer to the first fully connected layer!

We can think of the convolutional and pooling layers as creating useful representations of the data. Remember that both operations are local in the sense that they are taking into consideration windows of the data. Fully connected layers, in contrast, are global and connect every value in the previous max pooling layer ($\textbf{m}^{[13]}$) together.

The final step is to connect our last fully connected layer ($\textbf{a}^{[16]}$) to our output layer ($\hat{\textbf{y}}$). In order to make this transition, we have to use the softmax function, which is what we will discuss next.

## Softmax Layer

In order to make the final transition from fully connected to softmax layer, we use the softmax function. Let's discuss how the softmax function works next.

{% img /images/VGG_14.png [VGG_14]  %}

*It's difficult to see, but this is a diagram of the architecture of VGG-16 with example Softmax Layer $\textbf{a}^{[16]}$ and preceding Fully Connected Layer $\textbf{a}^{[15]}$ highlighted. Squint and look to the right.*

The transition from the fully connected layer $\textbf{a}^{[15]}$ to the softmax layer $\textbf{a}^{[16]}$ starts off as any fully connected layer usually does. We apply a matrix multiplication using $\textbf{W}^{[16]}$ and add a bias $\textbf{b}^{[16]}$ to attain $\textbf{z}^{[16]}$. $\textbf{W}^{[16]}$ has dimensions $(1000, 4096)$ and $\textbf{b}^{[16]}$ has dimensions $(100, 1)$, which makes sense, since $\textbf{a}^{[15]}$ is a row vector with dimensions $(4096, 1)$ and $\textbf{z}^{[15]}$ is a row vector with dimensions $(1000, 1)$.

$$\textbf{z}^{[16]} = \textbf{W}^{[16]}\textbf{a}^{[15]} + \textbf{b}^{[16]}$$

And this point, we would normally use a ReLU function to introduce nonlinearity. Instead, we are going to use the softmax function. This is similar to when we used the sigmoid function to produce the last fully connected layer in the previous post on [forward propagation](/forwardprop) in a fully connected neural network. We'll denote the softmax function with ($\sigma$). How do we compute the $ith$ element in $\textbf{a}^{[16]}$? We do the following:

$$a^{[16]}_{i,1} = \sigma_{(i, 1)}(\textbf{z}^{[16]})$$
$$a^{[16]}_{i,1} = \dfrac{e^{z^{[16]}_{i,1}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{j,1}}}$$

So in order to compute the $ith$ element of $a^{[16]}_{i,1}$, we take $e$ to the power of $z^{[16]}_{i,1}$ and divide it by the sum of $e$ to the power of all the elements in $\textbf{z}$. And after applying this activation function, we get a nice vector $\textbf{a}^{[16]}$ who's elements sum to $1$. Note that $\textbf{a}^{[16]}$ is equal to $\hat{\textbf{y}}$. Previously, we discussed how we wanted our output ($\hat{\textbf{y}}$) to be the probability that the training example image comes from one of $1,000$ classes. 

$$
\hat{\textbf{y}} =
\begin{bmatrix}
\hat{y}_{1} \\\\
\hat{y}_{2} \\\\ 
\vdots \\\\
\hat{y}_{1000}
\end{bmatrix}
$$

Where each $\hat{y}_{i}$ is equal to the probability that $y$ is equal to class $i$ given the input image $\textbf{x}$, or:

$$\hat{y}_{i} = P(y = i | \textbf{x})$$

After applying the softmax activation function, we now have a vector of probabilities who sum to 1.

So in the first part of this blog post, we broke down the different types of layers within VGG-16. We talked about conv layers, max pooling layers, flat layers, fully connected layers and finally the softmax layer that outputs the class probabilities. Since there is a lot of repetition within the model (which makes it appealing) we didn't go through each layer's dimensions and operations that are used to produce it. I want to wrap this section up by taking the architecture picture that I've used throughout this post, flip it, and label it with all the different types of layers we've discussed.

{% img /images/VGG_15.png [VGG_15]  %}

# Backpropagation for VGG16

Next up, I want to go into some detail about how backpropagation works for VGG-16. For each layer, our objective is to calculate two things: the partial derivative of the cost function $J$ with respect to that layer's activations and the partial derivative of the cost function with respect to the trainable parameters associated with that layer.

Before we start caluclating the partial derivatives for each example layer, let's talk about the cost function $J$.

## Understanding the Cost Function

In our previous post on [backprop](/backprop) our objective was to predict $y$, which could be either $0$ or $1$. Our prediction was a scalar $\hat{y}$.

For the Imagenet task however, our prediction $\hat{y}$ is a vector with dimensions $(1000, 1)$. Since we are using a softmax activation in our final layer, each value corresponds with the probability we think the training example belongs to the class label.

$$
\hat{\textbf{y}} =
\begin{bmatrix}
\hat{y}_{1} \\\\
\hat{y}_{2} \\\\ 
\vdots \\\\
\hat{y}_{i} \\\\ 
\vdots \\\\
\hat{y}_{1000}
\end{bmatrix}
$$

Where $\hat{y}_{i} = P(y = i | \textbf{x})$. $\hat{\textbf{y}}$ also has the attribute that the elements in it sum to $1$.

The loss function that we used for a single training example in our fully connected network was:

$$\mathcal{L}(\hat{y}, y) = -ylog(\hat{y}) - (1-y)log(1 -\hat{y})$$

And the loss function that we use for a single training example for VGG16 is very similar:

$$\mathcal{L}(\hat{\textbf{y}}, \textbf{y}) = -\sum_{i=1}^{1000} y_i log(\hat{y}_i)$$

Let's deconstruct what's happening in this loss function. Basically, for every possible image label (there are 1000 possible labels) we are calculating what is sometimes called the 'cross entropy' $y_i log(\hat{y}_i)$ between our prediction for that label $\hat{y}_i$ and the actual value $y_i$. If you look at the loss function right above it, it's essentially the same as the one directly preceding but for $2$ instead of $1000$ classes. We just choose to define the second class as being $1-y$, and our prediction as $1-\hat{y}$.

We want to minimize the loss function, which means we want to maximize the cross entropy $y_i log(\hat{y}_i)$. Recall that each image belongs to only $1$ of $1000$ classes. If our training example image was a dog and belonged to class $5$, $y_5 = 1$ and all other values ($y_1, ... y_4, y_6,...y_{1000}$) would be equal to $0$:

$$
\textbf{y} =
\begin{bmatrix}
0 \\\\
0 \\\\
0 \\\\
0 \\\\ 
1 \\\\ 
\vdots \\\\
0
\end{bmatrix}
$$

So the cross entropy for the value $y_5$ becomes $1 * log(\hat{y}_5)$ and the cross entropy for all other values become $0 * log(\hat{y}_i)$. And so, in order to maximize the cross entropy, we just need to maximize $\hat{y}_5$, which is the probability that given the data (pixels) of our training example image $\textbf{x}$ it belongs to class $5$. So it makes sense that this would be our loss function, since we want to maximize the probability that our training example comes from the correct class.

Where did this loss function come from? In our [fully connected network example](/backprop) we showed how we could derive our loss function for binary classification using the probability mass function for a Bernoulli distribution. We will take a similar approach and look at the probability mass function for a categorical (or multinoulli) distribution:

$$p(\textbf{y} | \hat{\textbf{y}}, \textbf{x}) = \prod_{i=1}^{1000} \hat{y}_i^{y_i}$$

This basically reads as the probability that given our example $\textbf{x}$ and prediction $\hat{\textbf{y}}$ we actually have an example with labels $\textbf{y}$ is equal to the product of $\hat{y}_i^{y_i}$ for each class label $i$. And if we take the log of both sides, we get:

$$log \ p(\textbf{y} | \hat{\textbf{y}}, \textbf{x}) = log \ \prod_{i=1}^{1000} \hat{y}_i^{y_i}$$
$$log \ p(\textbf{y} | \hat{\textbf{y}}, \textbf{x}) = \sum_{i=1}^{1000} log \ \hat{y}_i^{y_i}$$

And notice the right side is equal $-\mathcal{L}$:

$$log \ p(\textbf{y} | \hat{\textbf{y}}, \textbf{x}) = -\mathcal{L}(\hat{\textbf{y}}, \textbf{y}) $$

$$\mathcal{L}(\hat{\textbf{y}}, \textbf{y}) = -log \ p(\textbf{y} | \hat{\textbf{y}}, \textbf{x})$$

So when we say we want to minimize $\mathcal{L}$, we really mean we want to maximize the probability that $\textbf{y}$ is equal to it's value given our prediction $\hat{\textbf{y}}$ and feature vector $\textbf{x}$, given that we believe $\textbf{y}$ belongs to a categorical (or multillouni) distribution.

> Since log increases monotonically, maximizing $p(\textbf{y} | \hat{\textbf{y}}, \textbf{x})$ and maximizing log \ $p(\textbf{y} | \hat{\textbf{y}}, \textbf{x})$ is the same. 

So this is our loss function $\mathcal{L}$ which we use to determine how well our model is predicting the class label for a single training example $i$ given the pixels of the image $\textbf{x}^{(i)}$. But what if we were interested in how well our model was performing for a batch of $m$ training examples? We could take the average of our losses $\mathcal{L}$ over the $m$ training examples and call this our cost function, $J$:

$$J = \dfrac{1}{m}\sum^m_{i = 1} \mathcal{L(\hat{\textbf{y}}^{(i)},\textbf{y}^{(i)})}$$

We calculate the cost $J$ for our $m$ training examples, and then calculate two things: (1) the partial derivative of the cost function $J$ with respect to that layer's activations and (2) the partial derivative of the cost function with respect to the trainable parameters associated with that layer. We reshape (2) into gradients and then update the trainable parameters (which is essentially batch gradient descent) and use (1) to calculate the partial derivative of the cost function with respect to the trainable parameters in the previous layer (which is backpropagation).

For VGG16, we use a batch size of $m = 256$, so our cost function becomes:

$$J = \dfrac{1}{256}\sum^{256}_{i = 1} \mathcal{L(\hat{\textbf{y}}^{(i)},\textbf{y}^{(i)})}$$

Which simplifies to become:

$$J = -\dfrac{1}{256}\sum^{256}_{i = 1} \sum_{i=1}^{1000} y_i log(\hat{y}_i)$$

Great, so now we've defined our cost function. We now move to calculating the partial derivatives for each type of layer. Before moving forward I wanted to point something out. Let's say we are calculating the partial derivative of $J$ with respect to the weights in layer $j$:

$$\dfrac{\partial J}{\partial \textbf{W}^{[j]}} =  \dfrac{\partial}{\partial \textbf{W}^{[j]}} \dfrac{1}{256}\sum^{256}_{i = 1} \mathcal{L(\hat{\textbf{y}}^{(i)},\textbf{y}^{(i)})}$$

Notice that the differentiation on the right side can be placed inside the summation:

$$\dfrac{\partial J}{\partial \textbf{W}^{[j]}} =  \dfrac{1}{256}\sum^{256}_{i = 1} \dfrac{\partial\mathcal{L}}{\partial \textbf{W}^{[j]}}$$

So for every batch of 256 training examples $\textbf{x}^{[i]}$, we calculate the partial derivative with respect to the loss $\mathcal{L}$ for each trainable parameter, and then take the average of those 256 partial derivatives of the losses $\mathcal{L}$ to get the partial derivative of our cost function $J$ with respect to the trainable parameters.

Notice that we can calculate these 256 sets of partial derivatives in parallel, since they don't depend on each other. This is one of the ways we can parallelize backpropagation efficiently using GPUs.

In any case, instead of calculating the partial derivatives of $J$, we will just calculate the partial derivative of $\mathcal{L}$ for a single training example $\textbf{x}^{[i]}$ with the knowledge that we can just take the average of all the partial derivatives across the 256 examples in our batch to get the partial derivative of $J$.

Before moving forward, if you haven't checked out my blog post on [backprop](/backprop) it might be useful, since I'm using a lot of the same concepts (e.g. jacobian matrix, distinction between partial derivative and gradients)

## Backprop for the Softmax Layer

{% img /images/VGG_16.png [VGG_16]  %}

*Softmax layer is highlighted on the far right.*

The last layer of VGG-16 is a fully connected layer with a softmax activation. Since it is a fully connected layer, it has trainable parameters $\textbf{W}^{[16]}$ and $\textbf{b}^{[16]}$ and we therefore need to calculate the partial derivative of $\mathcal{L}$ with respect to both $\textbf{W}^{[16]}$ and $\textbf{b}^{[16]}$ which we can calculate the gradients as well as the partial derivative of $\mathcal{L}$ with respect to $\textbf{z}^{[16]}$, which can be used to backpropagate the error to the preceding layer:

$$\bigg ( d\textbf{W}^{[16]}, d\textbf{b}^{[16]}, \dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}} \bigg )$$

Let's focus our attention on calculating the partial derivative of $\mathcal{L}$ with respect to $\textbf{z}^{[16]}$. We can use the chain rule to rewrite this partial derivative as:

$$\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[16]}}\dfrac{\partial \textbf{a}^{[16]}}{\partial \textbf{z}^{[16]}}$$

So step $1$ is to figure out the partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[16]}$. Recall that the softmax layer $\textbf{a}^{[16]}$ is equal to our prediction for the class labels, $\hat{\textbf{y}}$
so we can write our loss function as:

$$\mathcal{L}(\textbf{a}^{[16]}, \textbf{y}) = -\sum_{i=1}^{1000} y_i log(a^{[16]}_{(i,1)})$$

Recall that $\mathcal{L}$ is a scalar value while $\textbf{a}^{[16]}$ is a column vector with dimensions $(1000, 1)$. The partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[16]}$ as represented as a Jacobian matrix is therefore $(1, 1000)$. 

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[16]}} = 
\begin{bmatrix}
\dfrac{\partial \mathcal{L}}{\partial a^{[16]}_{(1,1)}} &
\dfrac{\partial \mathcal{L}}{\partial a^{[16]}_{(2,1)}} &
\dots &
\dfrac{\partial \mathcal{L}}{\partial a^{[16]}_{(1000,1)}} 
\end{bmatrix}
$$

Notice that the first dimension of the Jacobian matrix is equal to the number of values in our output $\mathcal{L}$ which is $1$, and the second dimension is equal to the number of values in our input $\textbf{a}^{[16]}$, which is $1000$. We'll continue to use this formulation in the rest of the blog post.

Let's take the first value of the Jacobian, the partial derivative of $\mathcal{L}$ with respect to $a^{[16]}_{(1,1)}$. 

$$\mathcal{L}(\textbf{a}^{[16]}, \textbf{y}) = -\sum_{i=1}^{1000} y_i log(a^{[16]}_{(1,1)})$$

$$\mathcal{L}(\textbf{a}^{[16]}, \textbf{y}) = -y_1 log(a^{[16]}_{(1,1)})-y_2 log(a^{[16]}_{(2,1)}) \dots -y_1000 log(a^{[16]}_{(1000,1)})$$

$$\dfrac{\partial \mathcal{L}}{\partial a^{[16]}_{(1,1)}} = -\dfrac{y_1}{a^{[16]}_{(1,1)}} + 0 + \dots + 0$$
$$\dfrac{\partial \mathcal{L}}{\partial a^{[16]}_{(1,1)}} = -\dfrac{y_1}{a^{[16]}_{(1,1)}}$$

So applying this to every partial derivative in the Jacobian, we get:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[16]}} = 
\begin{bmatrix}
-\dfrac{y_1}{a^{[16]}_{(1,1)}} &
-\dfrac{y_2}{a^{[16]}_{(2,1)}} &
\dots &
-\dfrac{y_{1000}}{a^{[16]}_{(1000,1)}}
\end{bmatrix}
$$

Notice for any given training example $\textbf{x}$, its label $y$ will have a $1$ for its class and $0$ for all the others. So let's say for a random training example $\textbf{x}$ it has the label $3$, meaning $y_3 = 1$ and all the others are $0$. So the partial derivative will look like this:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[16]}} = 
\begin{bmatrix}
0 &
0 &
-\dfrac{y_3}{a^{[16]}_{(3,1)}} &
\dots &
0
\end{bmatrix}
$$

What this means is that we will only update the weights that relate to the third activation of the softmax layer, which makes sense, since we would only want to update the activation that corresponds with the true class label. We just calculated the partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[16]}$. Let's now work on the partial derivative of $\textbf{a}^{[16]}$ with respect to $\textbf{z}^{[16]}$. We'll start by analyzing it's Jacobian matrix:

$$ \dfrac{\partial \textbf{a}^{[16]}}{\partial \textbf{z}^{[16]}} = 
\begin{bmatrix}
\\
\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1, 1)}} &
\dotsc &
\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1000, 1)}} \\\\
\vdots & \ddots & \vdots \\\\
\dfrac{\partial a^{[16]}_{(1000,1)}}{\partial z^{[16]}_{(1, 1)}} &
\dotsc &
\dfrac{\partial a^{[16]}_{(1000,1)}}{\partial z^{[16]}_{(1000, 1)}} \\\\
\end{bmatrix}
$$

The partial derivative of $\textbf{a}^{[16]}$ with respect to $\textbf{z}^{[16]}$ has dimensions $(1000, 1000)$. Similar to before let's see if we can calculate the first value in the Jacobian Matrix, the partial derivative of $a^{[16]}_{(1,1)}$ with respect to $z^{[16]}_{(1,1)}$. We start with the formula of the softmax activation, that we defined in the previous post on [forward propagation](/vgg_forwardprop).

$$a^{[16]}_{(1,1)} = \sigma_{(1,1)}(\textbf{z}^{[16]})$$
$$a^{[16]}_{(1,1)} = \dfrac{e^{z^{[16]}_{(1,1)}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}}}$$
$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = \dfrac{\partial}{\partial z^{[16]}_{(1,1)}}\dfrac{e^{z^{[16]}_{(1,1)}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}}}$$

To calculate this partial derivative, we use the quotient rule for derivatives and get:

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = \dfrac{e^{z^{[16]}_{(1,1)}} * \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} - e^{2z^{[16]}_{(1,1)}}}{\big( \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} \big) ^ 2}$$

We can separate the term on the RHS into two terms:

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = \dfrac{e^{z^{[16]}_{(1,1)}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}}} - \dfrac{e^{2z^{[16]}_{(1,1)}}}{\big( \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} \big) ^ 2}$$

Substituting the softmax function $\sigma_{(1,1)}$ back in we get:

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = \sigma_{(1,1)}(\textbf{z}^{[16]}) - \sigma_{(1,1)}(\textbf{z}^{[16]})^2$$
$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = \sigma_{(1,1)}(\textbf{z}^{[16]})(1 - \sigma_{(1,1)}(\textbf{z}^{[16]}))$$

And using the fact that $a^{[16]}_{(i,1)} = \sigma_{(i,1)}(\textbf{z}^{[16]})$, we can simplify the partial derivative to become:

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(1,1)}} = a^{[16]}_{(1,1)}(1 - a^{[16]}_{(1,1)})$$

The partial derivative of $a^{[16]}_{(i,1)}$ with respect to $z^{[16]}_{(j,1)}$ looks like the calculation above when $i=j$. What happens to the partial derivative when $i \neq j$?

Let's take the example of the partial derivative of $a^{[16]}_{(1,1)}$ with respect to $z^{[16]}_{(2,1)}$.

$$a^{[16]}_{(1,1)} = \sigma_{(2,1)}(\textbf{z}^{[16]})$$
$$a^{[16]}_{(1,1)} = \dfrac{e^{z^{[16]}_{(2,1)}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}}}$$
$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(2,1)}} = \dfrac{\partial}{\partial z^{[16]}_{(2,1)}}\dfrac{e^{z^{[16]}_{(2,1)}}}{\sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}}}$$

To calculate this partial derivative, we use the quotient rule for derivatives and get:

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(2,1)}} = 
\dfrac{0 * \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} - e^{z^{[16]}_{(1,1)}}e^{z^{[16]}_{(2,1)}}}{\big( \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} \big) ^ 2}$$

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(2,1)}} = 
\dfrac{- e^{z^{[16]}_{(1,1)}}e^{z^{[16]}_{(2,1)}}}{\big( \sum_{j = 1}^{1000}e^{z^{[16]}_{(j,1)}} \big) ^ 2}$$

$$\dfrac{\partial a^{[16]}_{(1,1)}}{\partial z^{[16]}_{(2,1)}} = 
- a^{[16]}_{(1,1)}a^{[16]}_{(2,1)}$$

Generalizing this, we get:

$$\dfrac{\partial a^{[16]}_{(i,1)}}{\partial z^{[16]}_{(j,1)}} = 
\begin{cases}
   a^{[16]}_{(i,1)}(1 - a^{[16]}_{(i,1)})  & i = j \\\\
   - a^{[16]}_{(i,1)}a^{[16]}_{(j,1)} & i \neq j
\end{cases}
$$

And the Jacobian looks like this:

$$ \dfrac{\partial \textbf{a}^{[16]}}{\partial \textbf{z}^{[16]}} = 
\begin{bmatrix}
\\
a^{[16]}_{(1,1)}(1 - a^{[16]}_{(1,1)}) &
- a^{[16]}_{(1,1)}a^{[16]}_{(2,1)} & 
\dotsc &
- a^{[16]}_{(1,1)}a^{[16]}_{(999,1)} &
- a^{[16]}_{(1,1)}a^{[16]}_{(1000,1)} \\\\
\vdots & 
\vdots &
\ddots &
\vdots &
\vdots \\\\
- a^{[16]}_{(1000,1)}a^{[16]}_{(1,1)} & 
- a^{[16]}_{(1000,1)}a^{[16]}_{(2,1)} & 
\dotsc &
- a^{[16]}_{(1000,1)}a^{[16]}_{(999,1)} &
a^{[16]}_{(1000,1)}(1 - a^{[16]}_{(1000,1)}) \\\\
\end{bmatrix}
$$

The Jacobian has dimensions $(1000, 1000)$. Notice that the diagonal elements are equal to $a^{[16]}_{(i,1)}(1 - a^{[16]}_{(i,1)})$, whereas every other element is equal to $- a^{[16]}_{(i,1)}a^{[16]}_{(j,1)}$.

Substituting in our two Jacobian matricies, we get:

$$\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[16]}}\dfrac{\partial \textbf{a}^{[16]}}{\partial \textbf{z}^{[16]}}$$

$$\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}} = 
\begin{bmatrix}
-\dfrac{y_1}{a^{[16]}_{(1,1)}} &
-\dfrac{y_2}{a^{[16]}_{(2,1)}} &
\dots &
-\dfrac{y_{1000}}{a^{[16]}_{(1000,1)}}
\end{bmatrix}
\begin{bmatrix}
\\
a^{[16]}_{(1,1)}(1 - a^{[16]}_{(1,1)}) &
- a^{[16]}_{(1,1)}a^{[16]}_{(2,1)} & 
\dotsc &
- a^{[16]}_{(1,1)}a^{[16]}_{(999,1)} &
- a^{[16]}_{(1,1)}a^{[16]}_{(1000,1)} \\\\
\vdots & 
\vdots &
\ddots &
\vdots &
\vdots \\\\
- a^{[16]}_{(1000,1)}a^{[16]}_{(1,1)} & 
- a^{[16]}_{(1000,1)}a^{[16]}_{(2,1)} & 
\dotsc &
- a^{[16]}_{(1000,1)}a^{[16]}_{(999,1)} & 
a^{[16]}_{(1000,1)}(1 - a^{[16]}_{(1000,1)}) \\\\
\end{bmatrix}
$$

This matrix multiplication yields a Jacobian matrix with dimensions $(1, 1000)$. Let's look at the calculations for the first element of this matrix, the partial derivative of $\mathcal{L}$ with respect to $z^{[16]}_{(1,1)}$.

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
-\dfrac{y_1}{a^{[16]}_{(1,1)}}a^{[16]}_{(1,1)}(1 - a^{[16]}_{(1,1)}) + 
\dfrac{y_2}{a^{[16]}_{(2,1)}}a^{[16]}_{(2,1)}a^{[16]}_{(1,1)} +
\dfrac{y_3}{a^{[16]}_{(3,1)}}a^{[16]}_{(3,1)}a^{[16]}_{(1,1)} +  
\dots + 
\dfrac{y_{1000}}{a^{[16]}_{(1000,1)}}a^{[16]}_{(1000,1)}a^{[16]}_{(1,1)}
$$

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
-y_1(1 - a^{[16]}_{(1,1)}) + 
y_2a^{[16]}_{(1,1)} +
y_3a^{[16]}_{(1,1)} +  
\dots + 
y_{1000}a^{[16]}_{(1,1)}
$$

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
-y_1 + 
y_1a^{[16]}_{(1,1)}) + 
y_2a^{[16]}_{(1,1)} +
y_3a^{[16]}_{(1,1)} +  
\dots + 
y_{1000}a^{[16]}_{(1,1)}
$$

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
-y_1 + 
a^{[16]}_{(1,1)}\sum_i^{1000}{y_i}
$$

And recall because $\textbf{y}$ has only one class, one element is equal to $1$ whereas the others are equal to $0$. Therefore, the sum is equal to $1$.

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
-y_1 +
a^{[16]}_{(1,1)}*1
$$

$$\dfrac{\partial \mathcal{L}}{\partial z^{[16]}_{(1,1)}} = 
a^{[16]}_{(1,1)} - y_1
$$

Notice that the partial derivative is the same with just one class that we calculated in the previous [backprop](/backprop) blog post! All that fun work to get the same answer. [This is a great reference](https://www.ics.uci.edu/~pjsadows/notes.pdf) if you want more softmax backprop fun.

Generalizing to all elements in the Jacobian matrix, we get:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}} = 
\begin{bmatrix}
a^{[16]}_{(1,1)} - y_1 &
a^{[16]}_{(2,1)} - y_2 &
a^{[16]}_{(3,1)} - y_3 &
\dots &
a^{[16]}_{(1000,1)} - y_{1000} &
\end{bmatrix}
$$

And we end up with the partial derivative of $\mathcal{L}$ with respect to $\textbf{z}^{[16]}$ with dimensions $(1000, 1)$. We sometimes label the transpose of this partial derivative $\boldsymbol{\delta}^{[16]}$.

$$
\boldsymbol{\delta}^{[2]} = 
\bigg( \dfrac{\mathcal{L}}{\partial{\mathbf{z}^{[16]}}}\bigg)^T
$$

The dimensions of $\boldsymbol{\delta}^{[16]}$ are $(1000, 1)$, which match the dimensions of $\textbf{z}^{[16]}$. So we can think of $\boldsymbol{\delta}^{[16]}$ as the gradient for $\textbf{z}^{[16]}$, although we don't use this explicitly in gradient descent since $\textbf{z}^{[16]}$ has no updatable parameters.

Next up, we need to calculate the partial derivative of $\mathcal{L}$ with respect to both $\textbf{W}^{[16]}$ and $\textbf{b}^{[16]}$ in order to get the gradients $d\textbf{W}^{[16]}$ and $d\textbf{b}^{[16]}$ with gradient descent. Luckily, we've already calculated this in the previous post on [backprop](\backprop) so we can just use the results from that:

$$
\dfrac{\partial \textbf{z}^{[16]}}{\partial \textbf{W}^{[16]}} =
\begin{bmatrix}
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} &
0 &
0 &
0 &
\dots &
0 &
0 &
\dots &
\dots &
0 &
0 &
0 &
\dots &
0 &
0 \\\\
0 &
0 &
0 &
\dots &
0 &
0 &
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} &
\dots &
\dots &
0 &
0 &
0 &
\dots &
0 &
0 & \\\\
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots &
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots &
\dots &
\dots &
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots & \\\\
0 &
0 &
0 &
\dots &
0 &
0 &
0 &
0 &
0 &
\dots &
0 &
0 &
\dots &
\dots &
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} \\\\
\end{bmatrix}
$$

Notice that the Jacobian Matrix for the partial derivative of $\textbf{z}^{[16]}$ with respect to $\textbf{W}^{[16]}$ is a matrix with dimensions $(1000, 1000000)$. Since $\textbf{W}^{[16]}$ has dimensions $(1000, 1000)$, it has a total of $1000 * 1000 = 1,000,000$ weights which is represented in the second dimension of the Jacobian Matrix. 

Using the chain rule, the partial derivative of $\mathcal{L}$ with respect to $\textbf{W}^{[16]}$ is equal to:
$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{W}^{[16]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}}\dfrac{\partial \mathcal{\textbf{z}^{[16]}}}{\partial \textbf{W}^{[16]}}
\dfrac{\partial \mathcal{L}}{\partial \textbf{W}^{[16]}} = \boldsymbol{\delta}^{[16]T}\dfrac{\partial \mathcal{\textbf{z}^{[16]}}}{\partial \textbf{W}^{[16]}}
$$

Plugging in our two results, we get:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{W}^{[16]}} = 
\begin{bmatrix}
\delta^{[16]}_{(1,1)} &
\delta^{[16]}_{(2,1)} &
\delta^{[16]}_{(3,1)} &
\dots &
\delta^{[16]}_{(1000,1)}
\end{bmatrix}
\begin{bmatrix}
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} &
0 &
0 &
0 &
\dots &
0 &
0 &
\dots &
\dots &
0 &
0 &
0 &
\dots &
0 &
0 \\\\
0 &
0 &
0 &
\dots &
0 &
0 &
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} &
\dots &
\dots &
0 &
0 &
0 &
\dots &
0 &
0 & \\\\
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots &
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots &
\dots &
\dots &
\vdots &
\vdots &
\vdots &
\dots &
\vdots &
\vdots & \\\\
0 &
0 &
0 &
\dots &
0 &
0 &
0 &
0 &
0 &
\dots &
0 &
0 &
\dots &
\dots &
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
a^{[15]}_{(3,1)} &
\dots &
a^{[15]}_{(999,1)} &
a^{[15]}_{(1000,1)} \\\\
\end{bmatrix}
$$

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{W}^{[16]}} = 
\begin{bmatrix}
\delta^{[16]}_{(1,1)}a^{[15]}_{(1,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(1,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(1,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(1,1)} &
\delta^{[16]}_{(1,1)}a^{[15]}_{(2,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(2,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(2,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(2,1)} &
\dots &
\dots &
\delta^{[16]}_{(1,1)}a^{[15]}_{(1000,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(1000,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(1000,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(1000,1)} &
\end{bmatrix}
$$

So this is our Jacobian, with dimensions $(1,1000000)$. But we need our gradient matrix $d\textbf{W}^{[16]}$ to have dimensions that match $\textbf{W}^{[16]}$. So we will reshape the Jacobian into a $(1000, 1000)$ matrix.

$$
d\textbf{W}^{[16]} = 
\begin{bmatrix}
\delta^{[16]}_{(1,1)}a^{[15]}_{(1,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(1,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(1,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(1,1)} \\\\
\delta^{[16]}_{(1,1)}a^{[15]}_{(2,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(2,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(2,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(2,1)} \\\\
\vdots &
\vdots &
\vdots &
\dots &
\vdots \\\\
\delta^{[16]}_{(1,1)}a^{[15]}_{(1000,1)} &
\delta^{[16]}_{(2,1)}a^{[15]}_{(1000,1)} &
\delta^{[16]}_{(3,1)}a^{[15]}_{(1000,1)} &
\dots &
\delta^{[16]}_{(1000,1)}a^{[15]}_{(1000,1)} \\\\
\end{bmatrix}
$$

$$
d\textbf{W}^{[16]} = 
\begin{bmatrix}
\delta^{[16]}_{(2,1)} \\\\
\delta^{[16]}_{(2,1)} \\\\
\vdots \\\\
\delta^{[16]}_{(1000,1)}
\end{bmatrix}
\begin{bmatrix}
a^{[15]}_{(1,1)} &
a^{[15]}_{(2,1)} &
\vdots &
a^{[15]}_{(1000,1)}
\end{bmatrix}
$$

$$
d\textbf{W}^{[16]} = 
\boldsymbol{\delta}^{[16]}\textbf{a}^{[15]T}
$$

Ok great, now that we have $d\textbf{W}^{[16]}$ taken care of, let's move on to looking at the partial derivative of $\textbf{z}^{[16]}$ with respect to $\textbf{b}^{[16]}$. Again, we will use calculations we did in a simpler example in the [backprop](\backprop) post.

$$
\dfrac{\partial \textbf{z}^{[16]}}{\partial \textbf{b}^{[16]}} =
\begin{bmatrix}
1 &
0 &
\dots &
0 &
0 \\\\
0 &
1 &
\dots &
0 &
0 \\\\
0 &
0 &
\dots &
1 &
0 \\\\
0 &
0 &
\dots &
0 &
1 \\\\
\end{bmatrix}
$$

Using the chain rule, we get:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{b}^{[16]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[16]}}\dfrac{\partial \mathcal{\textbf{z}^{[16]}}}{\partial \textbf{b}^{[16]}}
\dfrac{\partial \mathcal{L}}{\partial \textbf{b}^{[16]}} = \boldsymbol{\delta}^{[16]T}\dfrac{\partial \mathcal{\textbf{z}^{[16]}}}{\partial \textbf{b}^{[16]}}
$$

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{b}^{[16]}} = 
\begin{bmatrix}
\delta^{[16]}_{(1,1)} &
\delta^{[16]}_{(2,1)} &
\delta^{[16]}_{(3,1)} &
\dots &
\delta^{[16]}_{(1000,1)}
\end{bmatrix}
\begin{bmatrix}
1 &
0 &
\dots &
0 &
0 \\\\
0 &
1 &
\dots &
0 &
0 \\\\
0 &
0 &
\dots &
1 &
0 \\\\
0 &
0 &
\dots &
0 &
1 \\\\
\end{bmatrix}
$$

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{b}^{[16]}} = 
\begin{bmatrix}
\delta^{[16]}_{(1,1)} &
\delta^{[16]}_{(2,1)} &
\delta^{[16]}_{(3,1)} &
\dots &
\delta^{[16]}_{(1000,1)}
\end{bmatrix}
$$

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{b}^{[16]}} = \boldsymbol{\delta}^{[16]T}
$$

So the partial derivative of $\mathcal{L}$ with respect to $\textbf{b}^{[16]}$ is just the transpose of $\boldsymbol{\delta}^{[16]}$, meaning that the gradient $d\textbf{b}^{[16]}$ will just be equal to:

$$d\textbf{b}^{[16]} = \boldsymbol{\delta}^{[16]T}$$

Great, so in this section we've talked about how to calculate the gradients $\boldsymbol{\delta}^{[16]}$, $d\textbf{W}^{[16]}$, and $d\textbf{b}^{[16]}$ and therefore know how to calculate the gradients for the softmax layer for hopefully any architecture we will encounter in the future.

After this softmax layer, we have two more fully-connected layers. The only difference between these two fully connected layers and the softmax layer we calculated above is that they use a ReLU as opposed to softmax activation function. I show how to calculate the gradients for these layers in my [backprop post](/backprop) that I'm sure you are sick of hearing about.

Working backwards, after the two fully connected layers we reshape our output from a column vector $f^{[13]}$ which has dimensions $(25088, 1)$ to a 3D tensor $m^{[13]}$ of shape $(7, 7, 512)$. What is the partial derivative for this transition? Since we don't change the dimensions and only reshape the elements, the partial derivative of $\textbf{f}^{[13]}$ with respect to $\textbf{m}^{[13]}$ is just the identity matrix with dimensions $(25088, 25088)$. So this partial derivative doesn't change the calculations for the calculating the partial derivatives for the preceding layer.

&nbsp;
## Backprop for the Max Pooling Layer

There are a total of $5$ max pooling layers in the VGG-16 architecture. Since they don't use trainable parameters, we only need to calculate their gradient $\boldsymbol{\delta}^{[13]}$, which is the input into the max pooling layer and we can use to calculate the gradients for the trainable parameters of that conv layer, $\textbf{W}_c^{[13]}$ and $\textbf{b}_c^{[13]}$

In order to calculate its gradient, we need to find the partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[13]}$. Using the chain rule, we get:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[13]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{m}^{[13]}}\dfrac{\partial \textbf{m}^{[13]}}{\partial \textbf{a}^{[13]}}
$$

Let's focus on the partial of $\textbf{m}^{[13]}$ with respect to $\textbf{a}^{[13]}$. $\textbf{m}^{[13]}$ has $7 * 7 * 512 = 25088$ values, and will therefore be the first dimension of the Jacobian. $\textbf{a}^{[13]}$ has $14 * 14 * 512 = 100352$, and is the second dimension of the Jacobian. Good lord, this Jacobian has dimensions $(25088, 100352)$ and has a total of $25088 * 100352 = 2517630976$ values. As you'll soon see, this Jacobian matrix is very sparse. There are computational shortcuts that libraries like tensorflow and pytorch use to handle these crazy matricies. So no worries.

Let's spend some time understanding how the elements of this matrix are aligned. We can start by looking at the first row of this matrix:

$$
\begin{bmatrix}
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(1, 1, 1)}}, & 
\dots, & 
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(14, 14, 512)}}
\end{bmatrix}
$$

So notice that the value in the numerator stays the same and the value of the denominator starts with $a^{[13]}_{(1, 1, 1)}$ and finishes at $a^{[13]}_{(14, 14, 512)}$. Since they are the ones chaning, let's focus on the values in the denominator:

$$
\begin{bmatrix}
a^{[13]}_{(1, 1, 1)} & 
\dots &
a^{[13]}_{(14, 14, 512)}
\end{bmatrix}
$$

So basically what we are doing is taking the 3D tensor array $\textbf{a}^{[13]}$ and flattening it into a 1D vector. In math, this operation is sometimes called $\textrm{vec}(\textbf{a})$. If we expanded this a little out:

$$
\begin{bmatrix}
a^{[13]}_{(1, 1, 1)} &
\dots &
a^{[13]}_{(14, 1, 1)} &
a^{[13]}_{(1, 2, 1)} &
\dots &
a^{[13]}_{(14, 14, 1)} &
a^{[13]}_{(1, 1, 2)} &
\dots &
a^{[13]}_{(14, 14, 512)}
\end{bmatrix}
$$

So we think of the first dimension as the width, the second as the height, and the third as the channel. So the first $14$ values of the vector are all the values for width for a height of $1$ and channel of $1$. Then we move to the next height $2$, keep the channel $1$ the same, and go through all $14$ values of the width. We continue this until we finish all $14$ heights, and we have the first $14 * 14 = 196$ values. Next, we reset the height to $1$ and width to $1$ and repeat the process for channel $2$. We go through all $512$ channels in this way, until we have a total of $14 * 14 * 512 = 100352$ values.

So that was the first row of our Jacobian matrix. We can think of the first column of our Jacobian matrix in a similar way:

$$
\begin{bmatrix}
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(1, 1, 1)}}, \\\\ 
\dots \\\\
\dfrac{\partial m^{[13]}_{(7,7,512)}}{\partial a^{[13]}_{(1, 1, 1)}}
\end{bmatrix}
$$

In this case, the value in the numerator is the one changing, from $m^{[13]}_{(1,1,1)}$ to $m^{[13]}_{(7,7,512)}$. We can think of this as being similar to above, where we convert $\textbf{m}^{[13]}$ into a flat vector, using the $vec$ operation.

Now that we have a better intutition about the values in the Jacobian Matrix, let's take the first value in this Jacobian Matrix and see if we can figure it out. Recall that the max pooling operation was defined as being:

$$m^{[13]}_{(i,j,k)} = \max_{i * s <= l < i * s + f, j * s <= l < j * s + f }a^{[2]}_{(l,m,k)}$$

Since all of our max pooling layers in VGG16 use a stride size of $2$ ($s = 2$) and a $2x2$ filter ($f = 2$), we get:

$$m^{[13]}_{(i,j,k)} = \max_{i * 2 <= l < i * 2 + 2, j * 2 <= l < j * 2 + 2 }a^{[2]}_{(l,m,k)}$$

Let's say we are interested in calculating $m^{[13]}_{(1,1,1)}$. We would look for the max value within a $2$ by $2$ window within $\textbf{a}^{[2]}$. Based on the equation above, the values we will look at are:

$$(a^{[13]}_{(1, 1, 1)}, a^{[13]}_{(2, 1, 1)}, a^{[13]}_{(1, 2, 1)}, a^{[13]}_{(2, 2, 1)})$$

Let's say the actual numerical values are as follows:

$$0, 5, 1, -4$$

Since $5$ is the largest, $m^{[13]}_{(1,1,1)} = a^{[13]}_{(2, 1, 1)}$, and the partial derivative of $m^{[13]}_{(1,1,1)}$ with respect to $a^{[13]}_{(2, 1, 1)}$ is equal to:

$$\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(1, 1, 1)}} = 1$$

Note that the partial derivatives of $m^{[13]}_{(1,1,1)}$ with respect to $a^{[13]}_{(1, 1, 1)}$, $a^{[13]}_{(1, 2, 1)}$, and $a^{[13]}_{(2, 2, 1)}$ are equal to $0$. And actually, the partial derivative of $m^{[13]}_{(1,1,1)}$ with respect to all the other values in the conv layer $\textbf{a}^{[13]}$ are also equal to 0 since they aren't in the 2 by 2 max pooling window we used. So the first row of our Jacobian matrix looks like this:

$$
\begin{bmatrix}
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(1, 1, 1)}}, &
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(2, 1, 1)}}, & 
\dots, & 
\dfrac{\partial m^{[13]}_{(1,1,1)}}{\partial a^{[13]}_{(14, 14, 512)}}
\end{bmatrix}
=
\begin{bmatrix}
0 &
1 &
\dots & 
0
\end{bmatrix}
$$

And we can continue this process for $m^{[13]}_{(2,1,1)}$ through $m^{[13]}_{(7,7,512)}$ and eventually fill up our $25088$ rows of our Jacobian matrix. And that's all to it! Notice that this matrix is very sparse. Each row has exactly one nonzero value and each column has at most one nonzero value. So we don't have to hold this whole matrix in memory. Instead, we can just record the locations of the nonzero values.

> Notice that this means that we will only update the weights with errors that correspond with the max values in each window.

We just figured out the partial derivative of $\textbf{m}^{[13]}$ with respect to $\textbf{a}^{[13]}$. Since we already calculated the partial derivative of $\mathcal{L}$ with respect to $\textbf{m}^{[13]}$, using the chain rule we can use that to calculate the partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[13]}$ and send that result to the preceding layer and for the first time calculate the partial derivatives of the convolutional weights and biases. We'll work on that next.

&nbsp;
## Backprop for the Conv Layer

The final type of layer that we need to calculate the partial derivatives for in order to get the gradients of the trainable parameters are conv layers. Let's focus on the first conv layer that we reach in VGG16, $\textbf{a}^{[13]}$. Our objective is to calculate the gradients for the trainable parameters in this layer, $d\textbf{W}^{[13]}_c$ and $d\textbf{b}^{[13]}_c$, as well as the partial derivative of $\mathcal{L}$ with respect to $\textbf{z}^{[13]}$, which we use to calculate the gradients for the preceding layers. Let's focus on this first. We can use the chain rule to break apart this partial derivative:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[13]}} = \dfrac{\partial \mathcal{L}}{\partial \textbf{a}^{[13]}}\dfrac{\partial \mathcal{\textbf{a}^{[13]}}}{\partial \textbf{z}^{[13]}}
$$

We've already calculated the partial derivative of $\mathcal{L}$ with respect to $\textbf{a}^{[13]}$, so we focus our attention on calculating the partial derivative of $\textbf{a}^{[13]}$ with respect to $\textbf{z}^{[13]}$. $\textbf{a}^{[13]}$ and $\textbf{z}^{[13]}$ have the same dimensions $(14, 14, 512)$. So therefore, the Jacobian matrix is $100352$ by $100352$. The position of the values from $\textbf{a}^{[13]}$ and $\textbf{z}^{[13]}$ is very similar to our previous max pooling example, where we can think of the changing values in each row as $vec(\textbf{z}^{[13]})$ and the changing values in each column as $vec(\textbf{a}^{[13]})$.

The transition from $\textbf{z}^{[13]}$ to $\textbf{a}^{[13]}$ just consists of applying the ReLU $g(z)$ nonlinear activation function to each element. What is the ReLU function?

$$
g(z) = \begin{cases}
   z &\text{if } z > 0  \\
   0 &\text{if } z =< 0
\end{cases}
$$

So the ReLU function just returns the value if it's greater than $0$, and 0 otherwise. What is the derivative of the ReLU function?

$$
g'(z) = \begin{cases}
   1 &\text{if } z > 0  \\
   \text{Undefined} &\text{if } z = 0  \\
   0 &\text{if } z < 0
\end{cases}
$$

Since this function is applied elementwise, the partial derivative of $\textbf{a}^{[13]}$ with respect to $\textbf{z}^{[13]}$ is just a diagonal matrix, with the derivatives of ReLU that correspond with that element in the diagonals and $0$ for all the other values.

$$
\dfrac{\partial \textbf{a}^{[13]}}{\partial \textbf{z}^{[13]}} = 
\begin{bmatrix}
g'(z^{[13]}_{(1,1,1)}) & 0 & \dots & 0 & 0 \\\\
0 & g'(z^{[13]}_{(2,1,1)}) & \dots & 0 & 0 \\\\
\vdots & \vdots & \ddots & \vdots & \vdots \\\\
0 & 0 & \dots & g'(z^{[13]}_{(13,14,512)}) & 0 \\\\
0 & 0 & \dots & 0 & g'(z^{[13]}_{(14,14,512)}) \\\\
\end{bmatrix}
$$

Great, so next we focus our attention on calculating the partial derivative of $\mathcal{L}$ with respect to $\textbf{W}_c^{[13]}$. Using the chain rule:

$$
\dfrac{\partial \mathcal{L}}{\partial \textbf{W}_c^{[13]}} = 
\dfrac{\partial \mathcal{L}}{\partial \textbf{z}^{[13]}} 
\dfrac{\partial \mathcal{\textbf{z}^{[13]}}}{\partial \textbf{W}^{[13]}}
$$

We just calculated the partial derivative of $\mathcal{L}$ with respect to $\textbf{z}^{[13]}$ and focus on calculating the partial derivative of $\textbf{z}^{[13]}$ with respect to $\textbf{W}_c^{[13]}$. This derivative takes it's first dimension from the values in $\textbf{z}^{[13]}$ $7 * 7 * 512 = 25088$ and second dimension from values in $\textbf{W}_c^{[13]}$ $3 * 3 * 512 * 512 = 2359296$. Its dimensions are therefore $(25088 , 2359296)$. Again, we can think of getting the values for each indexed partial derivatives using the $vec()$ function.

Let's deconstruct the derivative for $z^{[13]}_{(1,1,1)}$. Since this is from the first channel of $\textbf{z}^{[13]}$, we use the first filter channel in $\textbf{W}_c^{[13]}$. The calculation is as follows:

$$z^{[13]}_{(1,1,1)} =
s^{[12]}_{(1, 1, 1)}W^{[13]}_{c(1,1,1,1)} + 
s^{[12]}_{(2, 1, 1)}W^{[13]}_{c(2,1,1,1)} + 
s^{[12]}_{(3, 1, 1)}W^{[13]}_{c(3,1,1,1)} +
$$
$$ 
s^{[12]}_{(1, 2, 1)}W^{[13]}_{c(1,2,1,1)} + 
s^{[12]}_{(2, 2, 1)}W^{[13]}_{c(2,2,1,1)} + 
s^{[12]}_{(3, 2, 1)}W^{[13]}_{c(3,2,1,1)} +
$$
$$ 
s^{[12]}_{(1, 3, 1)}W^{[13]}_{c(1,3,1,1)} + 
s^{[12]}_{(2, 3, 1)}W^{[13]}_{c(2,3,1,1)} + 
s^{[12]}_{(3, 3, 1)}W^{[13]}_{c(3,3,1,1)} + \dots
$$

These are the calculations when we multiply the first channel of the first filter with the same padded input $\textbf{s^{[12]}}$. Next, we just move to the next channel of the first filter, and the third dimension goes from $1 \rightarrow 2$. We repeat this process for the 512 channels.

$$
\dots + 
s^{[12]}_{(1, 1, 2)}W^{[13]}_{c(1,1,2,1)} + 
s^{[12]}_{(2, 1, 2)}W^{[13]}_{c(2,1,2,1)} + 
s^{[12]}_{(3, 1, 2)}W^{[13]}_{c(3,1,2,1)} +
$$
$$ 
s^{[12]}_{(1, 2, 2)}W^{[13]}_{c(1,2,2,1)} + 
s^{[12]}_{(2, 2, 2)}W^{[13]}_{c(2,2,2,1)} + 
s^{[12]}_{(3, 2, 2)}W^{[13]}_{c(3,2,2,1)} +
$$
$$ 
s^{[12]}_{(1, 3, 2)}W^{[13]}_{c(1,3,2,1)} + 
s^{[12]}_{(2, 3, 2)}W^{[13]}_{c(2,3,2,1)} + 
s^{[12]}_{(3, 3, 2)}W^{[13]}_{c(3,3,2,1)} + \dots
$$

And then eventually we reach the channel number $512$ in the first filter:

$$
\vdots
$$
$$ 
s^{[12]}_{(1, 3, 512)}W^{[13]}_{c(1,3,512,1)} + 
s^{[12]}_{(2, 3, 512)}W^{[13]}_{c(2,3,512,1)} + 
s^{[12]}_{(3, 3, 512)}W^{[13]}_{c(3,3,512,1)} + b^{[1]}_{(1,1)}
$$

Recall from the blog post on [VGG16 forward propagation](/vgg_forwardprop) that $\textbf{s}^{[12]}$ is the activation from the previous layer padded with one border of zeros $p = 1$ using same padding.

What happens when we take the partial derivative of $z^{[13]}_{(1,1,1)}$ with respect to $W^{[13]}_{c(1,1,1,1)}$? Notice we just get the value for the padding layer $s^{[13]}_{(1, 1, 1)}$ and everything else is equal to 0. 

$$\dfrac{\partial z^{[13]}_{(1,1,1)}}{\partial W^{[13]}_{c(1,1,1,1)}} = s^{[12]}_{(1, 1, 1)} + 0 + 0 + ... + 0$$
$$\dfrac{\partial z^{[13]}_{(1,1,1)}}{\partial W^{[13]}_{c(1,1,1,1)}} = s^{[12]}_{(1, 1, 1)}$$

So notice that the first row of the Jacobian Matrix will have $3 * 3 * 512 = 4608$ nonzero elements, which correspond to the values multiplied by the weights in the filter. Notice that this is a very sparse row, since there are a total of $2359296$ elements in the row.

## Conclusion

In this post, we breakdown the architecture of VGG-16 and used it to explain some of the fundamental building blocks of the convolutional network - pooling layers and conv layers. We discussed some of the benefits of convolutional networks over fully connected layers and talked briefly about how backpropagation works for VGG-16.

You might have felt a little disatisfied with the math behind backpropagation the way that I explained it. At the end of this post, I also feel disatisfied. There is a transformation called `im2col` which flattens the input and filter bank as 2-dimensinoal matrices. Many explanations of backpropagation for convolutional networks use this function to simplify the computation (at the expense of memory) and I think it makes everything a lot simpler. In a future post, I will describe the `im2col` operation within the context of backpropagation, but I think for now we ware good.

As always, thanks so much for reading though my post! Any commens and questions would be greatly appreciated.

























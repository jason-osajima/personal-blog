title: The Math behind Neural Networks - Forward Propagation
slug: forwardprop
category: 
date: 2018-07-18
modified: 2018-07-18
tags: neural networks, machine learning

*This is part one in a two-part series on the math behind neural networks. Part one is about forward propagation. Part two is about backpropagation and can be found [here](/backprop).*

<!-- PELICAN_BEGIN_SUMMARY -->

When I started learning about neural networks, I found several articles and courses that guided you through their implementation in `numpy`. But when I started my research, I couldn't see past these basic implementations. In other words, I couldn't understand the concepts in research papers and I couldn't think of any interesting research ideas.

In order to go forward I had to go backwards. I had to relearn many fundamental concepts. The two concepts that are probably the most fundamental to neural networks are forward propagation and backpropagation. I decided to write two blog posts explaining in depth how these two concepts work. My hope is that by the end of this two part series you will have a deeper understanding of the fundamental underpinnings of both.

<!-- PELICAN_END_SUMMARY -->

I found three resources helpful. The first is the [Neural Network module](http://cs231n.github.io/neural-networks-1/) in the Stanford CS231n Convolutional Neural Networks for Visual Recognition course. The course materials are written by [Andrej Karpathy](http://karpathy.github.io). I enjoy reading Karpathy's work. He has a great conversational tone when describing concepts and it feels like you are traveling together on a roadtrip towards a better understanding of deep learning.

The second resource I would recommend is the [Deep Learning Book](http://www.deeplearningbook.org) written by Ian Goodfellow, Yoshua Bengio and Aaron Courville. It is an exahaustive resource of all the facts you will need to understand deep learning. It's on my to-do list to read and take notes on the entire book. In the meantime, I have used it frequently to learn about specific concepts that I wanted more information about.

The third is Andrew Ng's [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), available on Coursera. Ng has a special talent for explaining difficult ideas in simple ways. His ability to do this comes from his insistence on clear notation. He doesn't allow any notational detail to be lost. 

I recommend starting with the Stanford course and then moving on to Andrew Ng's course, all the while using the Deep Learning Book as a reference.

&nbsp;

### What is the problem we are trying to solve?

I probably don't have to convince you that neural networks have shown success in several [domains](http://www.wildml.com/2017/12/ai-and-deep-learning-in-2017-a-year-in-review/). In our example we will be focused on a binary classification problem.

A binary classifier is a supervised learning algorithm. We are given an input and our task is to predict which one of two classes the input belongs to. Each training example we use can be represented as $(\textbf{x}, y)$, where $\textbf{x} \in \mathbb{R}^{n_x}$ and $y \in (1, 0)$. If you aren't familiar with this notation, it just means that $\textbf{x}$ is a $n_x$-dimensional feature vector and $y$ can take on values $1$ or $0$.  Let's say we are trying to predict whether a person was happy ($0$) or sad ($1$) using features (1) how much sleep the person gets (2) how many times the person exercises in a week and (3) how many times the person hangs out with friends. Since we have three features, $n_x = 3$, and each of our $\textbf{x}$ training examples would be a $3$-dimensional vector. The values for each of the features can be represented with a subscript. So, $x_1$ would be the value for how much sleep a person gets, $x_2$ would be the value for how much a person exercises in a week, etc.

> A quick note on notation: For these blog posts, any time we define a vector or matrix, we will bold it. Anytime we define a scalar, we will keep it normal.

$m$ is equal to the number of training examples we have. So we end up with m pairs of training examples, and it can be written in this form:

$${(\textbf{x}^{(1)}, y^{(1)}), (\textbf{x}^{(2)}, y^{(2)}), (\textbf{x}^{(3)}, y^{(3)}),\ ..., (\textbf{x}^{(m)}, y^{(m)})}$$

Notice that we are using the superscript $(i)$ to denote the ith training example. So the third training example is $(\textbf{x}^{(3)}, y^{(3)})$. Next, we can take all of these $\textbf{x}^{(i)}$ vectors and line them up to create a matrix like so:

$$\textbf{X} = \begin{bmatrix}
    | & | & ... & | \\
   \textbf{x}^{(1)} & \textbf{x}^{(2)} & ... & \textbf{x}^{(m)} \\
   | & | & ... & |
\end{bmatrix}$$

The shape of $\textbf{X}$ is $(n_x, m)$, or $X \in \mathbb{R}^{n_x, m}$. Each row are the values for a given feature, and each column is a training example.

Similarly, we can group all of the output values $y$ for each training example into a vector:

$$\textbf{Y} = 
\begin{bmatrix}
y^{(1)} &
y^{(2)} &
... &
y^{(m)}
\end{bmatrix}$$

The shape of $\textbf{Y}$ is $(1, m)$ or $\textbf{Y} \in \mathbb{R}^{1, m}$.
&nbsp;

## Defining the Architecture

For the two blog posts, I decided to use a neural network with three layers. A ReLU activation function connects the input and two hidden layers and a sigmoid function connects the final hidden layer and the output layer.

There are a lot of great resources illustrating how forward propagation and backpropagation work for a one hidden-layer neural network or logistic regression, but I think the sweet spot for understanding both concepts occurs when you use a neural network with two hidden layers. So our example will focus on a three-layer hidden network. Here's what our lovely neural network looks like without labels.

{% img /images/nn_1.png [nn_1] %}

*Diagram of a Neural Network with Two Hidden Layers*

The first question that may come to mind is what does the output of the neural network represent? The input of a neural network is a feature vector from a training example ($\textbf{x}^{(i)}$). The output is our prediction $\hat{y}$. What does $\hat{y}$ represent?

Given a feature vector $\textbf{x}$, we want to predict whether the training example is a 1 or 0. We can think of our prediction as the probability that $y$ is equal to 1 given the feature vector $\textbf{x}$ for our training example, or $\hat{y} = P(y=1 | \textbf{x})$.

{% img /images/nn_2.png [nn_2] %}

*Diagram with input $\textbf{x}^{(i)}$ and output $\hat{y}^{(i)}$ for a given training example $i$.*

We define the first layer of the neural network as a feature vector from a given training example ($i$). In the diagram, each entry in the feature vector represents a scalar value. For example, $x^{(i)}_1$ is the value for the 1st feature for the ith training example.

Notice that the last layer of our neural network contains $\hat{y}^{(i)}$, which is our prediction for what we think the label should be for the ith training example.

Notice that our neural network has 4 layers of nodes, but we said in the beginning that our neural network has 2 hidden layers. What's the reasoning behind this? We treat our output and input layers as layers, so technically $x^{(i)} = a^{(i)[0]}$ and $a^{(i)[3]} = \hat{y}^{(i)}$. We define $a$ as a vector for the given layer, and the superscript $[j]$ tells us the layer number, so the input layer is the 0th layer and the output layer is the 3rd layer. Given that, how do we describe the hidden layers in between?

{% img /images/nn_3.png [nn_3] %}

*Diagram with hidden layers.*

In our diagram, we now have hidden layers $\textbf{a}^{(i)[1]}$ and $\textbf{a}^{(i)[2]}$, and output layer $\textbf{a}^{(i)[3]} = \hat{y}^{(i)}$ represented and our 3 layer hidden network is defined in our diagram. This confused me from the beginning because it's a 3-layer Neural Network with 2 hidden layers. So the number of hidden layers is number of layers - 1, since we count the output as a layer.

We can also vectorize our hidden layers the same way we vectorized our input ($\textbf{X}$) and output ($\hat{\textbf{Y}}$) by lining up the vectors for a hidden layer $j$:

$$\textbf{A}^{[j]} = \begin{bmatrix}
    | & | & ... & | \\
   \textbf{a}^{(1)[j]} & \textbf{a}^{(2)[j]} & ... & \textbf{a}^{(m)[j]} \\
   | & | & ... & |
\end{bmatrix}$$

This matrix $\textbf{A}^{[j]}$ becomes a $n^{\textbf{a}^{[j]}}$ by $m$ matrix, where $n^{a^{[j]}} =$ # of hidden units (or nodes) for layer j and $m$ is the number of training examples. Relating this back to our diagram, if $j = 1$, $n^{a^{[1]}} = 4$ and $\textbf{A}^{[1]}$ has the shape  $(4,m)$

What's interesting about the diagram (and something I didn't understand at first) is that it doesn't show any of the parameters for the model. The parameters are actually represented by the edges of the model. I'll talk about this next.

&nbsp;

## Going from layer to layer

Let's break down what's happening when we calculate $a_1^{(i)[1]}$, which is the first entry for the first hidden layer for the $ith$ training example.

{% img /images/nn_4.png [nn_4] %}

From the diagram, you can see that the input consists of all the entries from the previous layer (in this case the input layer from the $ith$ training example $\textbf{x}^{(i)}$ and the output is the entry for the first hidden layer $a_1^{(i)[1]}$.

{% img /images/nn_5.png [nn_5] %}

In order to calculate $a_1^{(i)[1]}$, we take each entry from $\textbf{x}^{(i)}$ and multiply it by a weight. The notation can be a little tricky, so let's break that down. Let's say we have $W^{[1]}_{13}$. We multiply this guy by the third entry in $\textbf{x}$, or $x^{(i)}_3$ to get the first entry in the $1st$ l ayer.

Let's breakdown the weights corresponding to $a_1^{(i)[1]}$ in our diagram. $\textbf{W}^{[1]}$ is a $(4, 3)$ matrix:

$$
\textbf{W}^{[1]} = 
\begin{bmatrix}
W^{[1]}_{11} & 
W^{[1]}_{12} &
W^{[1]}_{13} \\\\
W^{[1]}_{21} & 
W^{[1]}_{22} &
W^{[1]}_{23} \\\\
W^{[1]}_{31} & 
W^{[1]}_{32} &
W^{[1]}_{33} \\\\
W^{[1]}_{41} & 
W^{[1]}_{42} &
W^{[1]}_{43}
\end{bmatrix}
$$

The weights that we use to calculate $a_1^{(i)[1]}$ are the weights in the first row, or $\textbf{W}^{[1]}_{1-}$. $\textbf{W}^{[1]}_{1-}$ is a $(1, 3)$ row vector:

$$
\textbf{W}^{[1]}_{1-} = 
\begin{bmatrix}
W^{[1]}_{11} & 
W^{[1]}_{12} &
W^{[1]}_{13}
\end{bmatrix}
$$

We can then multiply this vector by $\textbf{x}^{(i)}$ and we get a nicer, compact representation:

{% img /images/nn_6.png [nn_6] %}

Note that we add a bias $b^{[1]}_{1}$ to $\textbf{W}^{[1]}_{1-}\textbf{x}^{(i)}$. The bias are other parameters besides the weights that our model learns. Why do we add a bias? In order to answer that, let's talk about our activation function $g()$.

Different neural network architectures make different choices for activation functions, but in our example to keep it simple we will use a rectified linear unit, or ReLU function.

The ReLU function is defined as the following:

$$
g(z) = \begin{cases}
   z &\text{if } z > 0  \\
   0 &\text{otherwise}
\end{cases}
$$

What is the role of activation functions in Neural Networks? Activation functions introduce non-linearity into the neural network. Without activation functions, neural networks would simplify to linear functions. Let's see how that works with respect to our example. If we simplified our neural network by taking out the bias terms and activation functions, the neural network becomes:

$$\hat{y}^{(i)} = \textbf{W}^{[2]}\textbf{W}^{[1]}\textbf{A}^{(i)[0]}$$

Notice that if we multiply a matrix of weights ($\textbf{W}^{[2]}$) by another matrix of weights ($\textbf{W}^{[1]}$), we get one matrix of weights ($\textbf{W} = \textbf{W}^{[2]}\textbf{W}^{[1]}$). So our example simplifies to a linear function:

$$\hat{y}^{(i)} = \textbf{W}\textbf{A}^{(i)[0]}$$


Now back to our discussion about the bias term. We add a bias term in order to shift our activation function (in our case, the ReLU) to the left or right, which is usually important for learning because it makes the model more flexible.

&nbsp;

## Forward propagation in a 3-layer Network

Now that we discussed some of the elements of a 3-layer network, let's (finally) introduce the concept of forward propagation. 

Forward propagation is basically the process of taking some feature vector $\textbf{x}^{(i)}$ and getting an output $\hat{y}^{(i)}$. Let's breakdown what's happening in our example.

{% img /images/nn_8.png [nn_8] %}

As you can see, we take a (3 x 1) training example $\textbf{x}^{(i)}$, get the (4 x 1) activations from the first hidden layer $\textbf{a}^{(i)[1]}$. Next, we get the (1 x 2) activations from the second hidden layer $\textbf{a}^{(i)[2]}$ and the final (1 x 1) output $\hat{y}^{(i)}$. As we mentioned earlier, $\hat{y}^{(i)}$ is the probability that $y^{(i)}$ is of the positive class given the information we know in the form of $\textbf{x}^{(i)}$, or $\hat{y}^{(i)} = P(y^{(i)} = 1 | x^{(i)})$. In summary, forward propagation looks like this:

$$\textbf{x}^{(i)} \rightarrow \textbf{a}^{(i)[1]} \rightarrow \textbf{a}^{(i)[2]} \rightarrow \hat{y}^{(i)}$$

Next, let's discuss the inner-workings of each of these transitions.

&nbsp;
### Input $\rightarrow$ 1st Hidden Layer ($\textbf{x}^{(i)} \rightarrow \textbf{a}^{(i)[1]}$)

First, what's happening when we transition from our vector of features for the first training example $i$ to the activations from our first hidden layer. We start by multiplying $\textbf{x}^{(i)}$ by the weights and bias of the first hidden layer, $\textbf{W}^{[1]}$ and $\textbf{b}^{[1]}$ to get $\textbf{z}^{(i)[1]}$. People sometimes call $\textbf{z}^{(i)[1]}$ the activity of the hidden layer 1 for training example $i$.

$$\textbf{z}^{(i)[1]} = \textbf{W}^{[1]}\textbf{x}^{(i)} + \textbf{b}^{[1]}$$

So we start by multiplying $\textbf{x}^{(i)}$ by $\textbf{W}^{[1]}$. $\textbf{x}^{(i)}$ is a (3 x 1) matrix, and $\textbf{W}^{[1]}$ is a (4 x 3) matrix. We then add the bias, $\textbf{b}^{[1]}$. The dimensions of the bias $\textbf{b}^{[1]}$ match the dimensions of $\textbf{z}^{(i)[1]}$ which are (4, 1). Once we get the activity matrix $\textbf{z}^{(i)[1]}$, we apply the activation function to each element in $\textbf{z}^{(i)[1]}$. Recall that the activation function that we chose is ReLU, which we defined as:

So we get:

$$\textbf{a}^{(i)[1]} = g(\textbf{z}^{(i)[1]})$$

Which just indicates that the ReLU function $g()$ is applied elementwise to $\textbf{z}^{(i)[1]}$ to get $\textbf{a}^{(i)[1]}$.

$\textbf{a}^{(i)[1]}$ has the same dimensions as $\textbf{z}^{(i)[1]}$, so it's (4,1).

&nbsp;
### 1st Hidden Layer $\rightarrow$ 2nd Hidden Layer ($\textbf{a}^{(i)[1]} \rightarrow \textbf{a}^{(i)[2]}$)

This section is going to be almost identical to the previous section. We start by multiplying $\textbf{a}^{(i)[1]}$ by $\textbf{W}^{[2]}$. $\textbf{a}^{(i)[1]}$ is a (4 x 1) matrix, and $\textbf{W}^{[2]}$ is a (2 x 4) matrix. We then add the bias, $\textbf{b}^{[2]}$. The dimensions of the bias $\textbf{b}^{[2]}$ match the dimensions of $\textbf{z}^{(i)[2]}$ which are (2, 1). Once we get the activity matrix $\textbf{z}^{(i)[2]}$, we apply the ReLU activation function to each element in $\textbf{z}^{(i)[2]}$. So we get:

$$\textbf{a}^{(i)[2]} = g(\textbf{z}^{(i)[2]})$$

$\textbf{a}^{(i)[2]}$ has the same dimensions as $\textbf{z}^{(i)[2]}$, so it's (2,1).

&nbsp;
### 2nd Hidden Layer $\rightarrow$ Output ($\textbf{a}^{(i)[2]} \rightarrow \hat{y}^{(i)}$)

So now we have $\textbf{a}^{(i)[2]}$. We again start by multiplying $\textbf{a}^{(i)[2]}$ by $\textbf{W}^{[3]}$. $\textbf{a}^{(i)[2]}$ is a (2 x 1) matrix, and $\textbf{W}^{[3]}$ is a (1 x 2) matrix. We then add the bias, $b^{[3]}$. The dimensions of the bias $b^{[3]}$ match the dimensions of $z^{(i)[3]}$ which are (1, 1). Once we get the activity matrix $z^{(i)[3]}$, we apply the activation function to each element in $z^{(i)[3]}$. Since this is the final layer of our neural network, we will use a sigmoid activation function:

$$
\sigma(z) = \dfrac{1}{1+e^{-z}}
$$

The sigmoid activation function is rarely used in modern neural networks because it suffers from the vanishing gradient problem, but it is often used as the final activation function before the output. The reason is that it is able to squash values to be between 0 and 1, which is what we want since recall we want $\hat{y}^{(i)}$ to be between 0 and 1 since $\hat{y}^{(i)} = P(y^{(i)} = 1 | x^{(i)})$.

So we get:

$$\hat{y}^{(i)} = a^{(i)[3]} = \sigma(z^{(i)[3]})$$

$\hat{y}^{(i)}$ has the same dimensions as $z^{(i)[3]}$, so it's (1,1).

And that's it!

&nbsp;
### Conclusion

In this blog post, I used a 3-layer neural network example to help us deconstruct the math involved in forward propagation. One of the hardest parts of this process was making sure the dimensions of all the matricies match up, so some parting thoughts on dimensions:

- If you think about just one training example $i$ like we did, the dimensions of activations $\textbf{a}$ will always be $(n_{a}, 1)$, where $n_a$ is equal to the number of nodes in the layer. So for example, if we had 100 nodes in the 5th hidden layer, $\textbf{a}^{(i)[5]}$ would have dimensions (100, 1).
- If you think about $m$ training examples, you simply switch the 2nd dimension from $1$ to $m$. So for example, if we had 100 nodes in the 5th hidden layer, for m-training examples $\textbf{a}^{(i)[5]}$ would have dimensions (100, m).
- The weights $\textbf{W}^{l}$ for layer $l$ will have dimensions $(n_{a}^{[l]}, n_{a}^{[l-1]})$ Notice that the weights don't care about the second dimension of activations $\textbf{a}^{[l]}$, they just care about that 1st dimension.
- The final output layer is also our $\hat{y}^{(i)}$. We count this layer when we label a neural network along with the hidden layers, so a 3-layer Neural Network will only have 2 hidden layers. 

Now that we have the forward propagation figured out, we can generate a prediction $\hat{y}^{(i)}$ given a feature vector for the ith training example $\textbf{x}^{(i)}$. But is this a good prediction? How does it compare to the actual label, $y^{(i)}$? In order to come up with a good prediction not just for the ith training example but for all examples, we need an algorithm to find the best values for our weights $\textbf{W}$ and biases $\textbf{b}$. 

The most popular algorithm to use is called backpropagation, which we will discuss in the [next post](/backprop).







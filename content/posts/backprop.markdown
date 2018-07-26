title: The Math behind Neural Networks - Backpropagation
slug: backprop
category: 
date: 2018-07-18
modified: 2018-07-18
tags: neural networks, machine learning

*This is part two in a two-part series on the math behind neural networks. Part two is about backpropagation. Part one is about forward propagation and can be found [here](/forwardprop).*

## Introduction

<!-- PELICAN_BEGIN_SUMMARY -->

The hardest part about deep learning for me was backpropagation. Forward propagation made sense; basically you do a bunch of matrix multiplications, add some bias terms, and throw in non-linearities so it doesn't turn into one large matrix multiplication. Gradient descent also intuitively made sense to me as well; we want to use the partial derivatives of our parameters with respect to our cost function ($J$) to update our parameters in order to minimize $J$. 

The objective of backpropagation is pretty clear: we need to calculate the partial derivatives of our parameters with respect to cost function ($J$) in order to use it for gradient descent. The difficult part lies in keeping track of the calculations, since each partial derivative of parameters in each layer rely on inputs from the previous layer. Maybe it's also the fact that we are going backwards makes it hard for my brain to wrap my head around it.

<!-- PELICAN_END_SUMMARY -->
&nbsp;
## Motivation

Why do we need to learn the math behind backpropagation? Today we have great deep learning frameworks like Tensorflow and PyTorch that do backpropagation automatically, so why do we need to know how it works under the hood?

Josh Waitzkin is an internationally-recognized chess player (and the movie and book "Searching for Bobby Fischer" were both based on his career) and world champion in Tai Chi Chuan. 

Most kids start their chess career by learning opening variations of the game. Kids learn strong opening positions in order to gain a decisive advantage in the beginning of a match. This advantage proves too much for most of their opponents and they end up winning. 

Josh Waitzkin's teacher did not focus on opening variations. Instead, he focused on end game scenarios, such as King v. King or King v. King Pawn. These end game scenarios were simple enough for Josh to develop an intuitive understanding of how pieces interact with one another. 

Kids that learn opening variations couldn't internalize these first principles about the interactions between pieces because there was too much stuff going on. 

> A critical challenge for all practical martial artists is to make their diverse techniques take on the efficiency of the jab. When I watched William Chen spar, he was incredibly understated and exuded shocking power. While some are content to call such abilities chi and stand in awe, I wanted to to understand what was going on. The next phase of my martial growth would involve turning the large into the small. My understanding of this process, in the spirit of my numbers to leave numbers method of chess study, is to touch the essence (for example, highly refined and deeply internalized body mechanics or feeling) of a technique, and then to incrementally condense the external manifestation of the tecnhique while keeping true to its essence. Over time expanding decreases while potency increases. I call this method 'making smaller circles'.

I think there are several ways that we can interpret this quote. For someone who is a machine learning practitioner, it may be good to take a model that you have a pretty good understanding of and try and simplify it to try to understand everything at its most basic level.

The best resource for me to learn backpropagation has been Andrew Ng's [Deep Learning Specialization in Coursera]((https://www.coursera.org/specializations/deep-learning). He is always very clear about understanding the intutition behind the problem, defining the problem and then laying out the mathematical notation. In this blog post I will rely heavily on the notation and concepts used in his course.

Let's make smaller circles.

Backpropagation can be used in different ways, but for our purposes we will use it to train a binary classifier. In my [previous post](/forwardprop) on forward propagation, I layout the architecture for a 3 layer Neural Network, which we will rely on for this example.

Backpropagation starts with our loss function, so we will introduce this idea first. But before we get into the math, let's define what notation we will use through the course of this blog post.

&nbsp;
## Some Notation

For this blog post, any time we define a vector or matrix, we will bold it. Anytime we define a scalar, we will keep it normal. In the previous blog post on forward propagation, we introduced a 3-layer neural network architecture:

{% img /images/nn_3.png [nn_3] %}

So for example, $\textbf{a}^{(i)[1]}$ is a vector and is therefore bolded. The third entry for $\textbf{a}^{(i)[1]}$ is $a^{(i)[1]}_{31}$. Since it is a scalar, it is not bolded. 

You may have noticed that $a^{(i)[3]}$ and $\hat{y}^{(i)}$ are not bolded. That's because they are scalars $a^{(i)[3]} = \hat{y}^{(i)} \in (0, 1)$, and represent the probability that we think the $ith$ example belongs to the positive class, $y = 1$. More on this later.

The $i$ denotes that $\textbf{a}^{(i)[1]}$ is the activation in the $L = 1$ layer for the $ith$ example. For simplicity, we will get rid of the $(i)$ notation and assume that we are working with the $ith$ training example. Our architecture therefore becomes:

{% img /images/nn_simplified.png [nn_simplified] %}


Lovely, that looks much simpler. You might be wondering why we decided to define all of our vectors as column vectors instead of row vectors. If a vector has $m$ entries, a column vector is defined to be a $(m,1)$ dimensional matrix and a row vector is a $(1, m)$ dimensional matrix.

Some people use row vectors and others use column vectors. Most of the resources I used to write this blog post use column vectors to define the activations for each layer, so that's what we will roll with.

When we define a column vector for the outputs from different layers, we will bold it and use a lowercase letter to represent it. When we define a weight matrix, we will bold it and use an uppercase letter to define it. So for example, the weight matrix we use to transition from the 2nd layer (1st hidden layer) to the 3rd layer (2nd hidden layer) would be $\textbf{W}^{[2]}$ with dimensions $(2, 4)$ that match the 1st dimension of the layer it's transitioning to (2) and the 1st dimension of the layer it comes from (4). Each entry is a scalar, and therefore is not bolded.

$$\textbf{W}^{[2]} = 
\begin{bmatrix}
    W^{[2]}_{11} & W^{[2]}_{12} & W^{[2]}_{13} & W^{[2]}_{14} \\\\
    W^{[2]}_{21} & W^{[2]}_{22} & W^{[2]}_{23} & W^{[2]}_{24} \\
\end{bmatrix}$$

&nbsp;
## Understanding the Loss Function

In the previous post we used forward propagation to go from an input vector for the $ith$ training example $\textbf{x}$ to $\hat{y}$. Recall that $\hat{y}$ is our best guess for the class $\textbf{x}$ belongs to. In our example, $\textbf{x}$ could belong to either happy ($0$) or sad ($1$).  $y$ is the class (either 0 or 1) that $\textbf{x}$ actually belongs to. So how can we measure how well our model is doing, i.e. how can we measure how close the prediction $\hat{y}$ is to the actual $y$?

In order to measure the error between these two values, we use what's called a loss function. When I was introduced to the concept of a loss function, I immediately thought we should use this one:

$$\mathcal{L}(\hat{y}, y) = \dfrac{1}{2}(\hat{y} - y)^2$$

This is a pretty simple loss function: just subtract the actual from the predicted, square it so it isn't negative, and then divide it by 2 (so the derivative looks prettier). It turns out that people don't use this loss function in logistic regression because when you try to learn the parameters the optimization problem is non-convex.

A better loss function to use is this:
$$\mathcal{L}(\hat{y}, y) = -ylog(\hat{y}) - (1-y)log(1 -\hat{y})$$

There are several choices for what we can use for our loss function. Let's spend some time to understand how we use this loss function. 

> One thing that I didn't get when I first started working on this: In high school math if you had for example $log \ 2$, this was shorthand for log base 10, or $log_{10} \ 2$. However, you'll find that most people that work on stats and computer science problems actually use $log \ 2$ to mean $ln \ 2$. So in this blog post, whenever I use $log$, I mean $ln$.

Our objective is to try to get $\mathcal{L}$ to be as low as possible, since $\mathcal{L}$ represents the error between our prediction $\hat{y}$ and the actual $y$. Notice that when $y = 1$, our equation turns into this:

$$\mathcal{L}(\hat{y},  y = 1) = -1log(\hat{y}) - (1-1)log(1 - \hat{y})$$

$$\mathcal{L}(\hat{y},  y = 1) = -1log(\hat{y})$$

So to minimize $\mathcal{L}$ we want $\hat{y}$ to be as large as possible, which makes sense since in the final layer we put each entry $z^{[3]}_{1j}$ in the activity $\mathbf{z}^{[3]}$ through the sigmoid function, like $\sigma(\mathbf{z}^{[3]})$. The range of the sigmoid function is $(0, 1)$, so the greatest value that $\hat{y}$ can take is a number super close to 1. Conversely, when $y = 0$, our equation turns into this:

$$\mathcal{L}(\hat{y},  y = 0) = -0log(\hat{y}) - (1-0)log(1 - \hat{y})$$
$$\mathcal{L}(\hat{y},  y = 0) = -1log(1 - \hat{y})$$

In this case, in order to minimize $\mathcal{L}$, we want $\hat{y}$ to be close to 0 as possible.

This seems like it works, but where does this loss function come from? I'm glad you asked, it's kind of fun to figure out how it works.

So in the previous post we talked about how $\hat{y}$ is the probability that the example $\mathbf{x}$ is from either the class 1, or $y = 1$. And also keep in mind that all of these technically should have a superscript $(i)$ attached to them. So more formally, $\hat{y} = P(y = 1 | \mathbf{x})$.

We can think of $y$ as a Bernoulli random variable that can take on 1 with probability $\hat{y}$ and 0 with probability $1 - \hat{y}$. 

The probability mass function for a Bernoulli random variable looks like this:

$$p(k | p) = k^p(1-k)^{(1-p)}$$

Which tells you the probability that $k$ is equal to a particular value. We want to calculate the probability that $y$ takes on a particular value so, thinking about our example, we get:

$$p(y | \hat{y}, \mathbf{x}) = y^{\hat{y}}(1-y)^{(1-\hat{y})}$$

We want to maximize $p$, or maximize the probability that given our training example feature vector $\mathbf{x}$ and prediction outputted from our neural network $\hat{y}$, we get the value $y$.

Ok so hopefully that makes sense so far. Why can't we just use this as our loss function, since the function shows how close how our prediction ($\hat{y}$) is to our actual ($y$)? The reason is in backpropagation we need to take the derivative of our loss function, and it gets a little messy to take the derivative of this function.

But notice that if we take the log of both sides, our objective stays the same. Instead of maximizing $p$, we still just need to maximize the $log$ of $p$, since the $log$ function increases monotonically. If we take the $log$ of both sides, and do a little math, we get:

$$log \ p(y | \hat{y}, \mathbf{x}) = log \ (y^{\hat{y}}(1-y)^{(1-\hat{y})})$$
$$log \ p(y | \hat{y}, \mathbf{x}) = \hat{y}log \ y + (1-\hat{y})log \ (1-y)$$

Notice that the left side just becomes $-\mathcal{L}(\hat{y}, y)$.

$$log \ p(y | \hat{y}, x) = -\mathcal{L}(\hat{y}, y)$$
$$\mathcal{L}(\hat{y}, y) = -log \ p(y | \hat{y}, x)$$

So when we say we want to minimize $\mathcal{L}$, we really mean we want to maximize the probability that $y$ is equal to it's value given our prediction $\hat{y}$ and feature vector $x$. 

So we figured out what $\mathcal{L}$ is equal to, which represents our loss for one training example $i$. We could just use the gradients of $\mathcal{L}$ with respect to each scalar entry in each of our parameters. We then could use those gradients to update the values of our parameters in gradient descent. In that case, our loss function $\mathcal{L}$ would be the same as our cost function $J$ or:

$$ J = \mathcal{L}(\hat{y},y) $$

In this case, we call our optimization algorithm stochastic gradient descent. We could also take a batch of training examples, say $m$ training examples and define our cost function $J$ to be the average of the loss $\mathcal{L}(\hat{y}^{(i)},y^{(i)})$ for $m$ training examples, or:

$$J = \dfrac{1}{m}\sum^m_{i = 1} \mathcal{L(\hat{y}^{(i)},y^{(i)})}$$

If $m$ is equal to the number of training examples we have access to, we usually call our optimization algorithm batch gradient descent. If $m$ is less than the number of training examples, we call it mini-batch gradient descent.

For simplicity, in this blog post we will focus on stochastic gradient descent, so our cost function $J$ is:


$$ J = \mathcal{L}(\hat{y},y) $$

Great, so now we have defined $J$ and we want to minimize it. How do we do that? Most people use gradient descent, which requires us to calculate the gradient for $J$ with respect to the parameters that we can change. Calculating these gradients is the objective of backpropagation.

&nbsp;
## Introducting Backpropagation

In backpropagation, our objective is to calculate the gradients of $\mathcal{L}$ with respect to what we can change in our neural network. In our three layer network, we can change the value of our parameters. Recall that the architecture of our network looked like this:

{% img /images/nn_3.png [nn_3] %}

We can think of the nodes from the two layers connecting the input to the output layer as the intermediate products of the model.

Notice that this diagram doesn't include any of the parameters of our model. The parameters are the weights ($\textbf{W}^{[j]}$) and biases ($\textbf{b}^{[j]}$) associated with the j-th layer. Because in order to go from one layer to the next, we multiply the nodes from the previous layer by the weights, add a bias, and send it through an activation function.

We can think of the lines that connect the nodes of each layer to represent these transformations. In the diagram, there are three sets of lines connecting the four layers, and unsurprisingly there are three sets of weights and biases to go along with them:

$$(\textbf{W}^{[1]}, \textbf{b}^{[1]}, \textbf{W}^{[2]}, \textbf{b}^{[2]}, \textbf{W}^{[3]}, b^{[3]})$$

Let's understand the dimensions of each of these parameters. In the previous post, we talked about how the dimensions of the weights that connect layers are equal to the number of entries in those layers, represented by column vectors. So for example, $\textbf{W}^{[1]}$ connects $\textbf{x}$ a column vector with 3 entries to $\textbf{z}^{[1]}$, a column vector with 3 entires. So the dimensions of $\textbf{W}^{[1]}$ will be $(4, 3)$. Similarly, the dimensions of $\textbf{W}^{[2]}$ will be $(2, 4)$ and the dimensions of $\textbf{W}^{[3]}$ will be $(1, 2)$.

Biases are simpler, since they match the dimensions of the layer that we are headed towards. So for example, $\textbf{b}^{[1]}$ is $(4, 1)$, $\textbf{b}^{[2]}$ is $(2, 1)$, and $b^{[3]}$ is a scalar value.

&nbsp;
### Gradients for Activation Functions?

Do we need to worry about parameters in the activation functions we use? Let's first recall the activation functions that we are using in our example. We use a ReLU function $g()$ to go from the input layer $\textbf{x}$ to the first hidden layer $\textbf{a}^{[1]}$, a ReLU function $g()$ to go from the first hidden layer $\textbf{a}^{[1]}$ to the second hidden layer $\textbf{a}^{[2]}$.

$$
g(z) = \begin{cases}
   x &\text{if } z > 0  \\
   0 &\text{otherwise}
\end{cases}
$$

Where $z$ is a scalar value.

Notice that the ReLU function doesn't include any parameters that we would need to optimize in our model.

We use a sigmoid function $\sigma()$ to go from the second hidden layer $\textbf{a}^{[2]}$ to the final output layer $\hat{y}$. 

$$
\sigma(z) = \dfrac{1}{1+e^{-z}}
$$

Same as the ReLU function, there are no parameters that we need to optimize in this function.

So we talked about how in backpropagation we calculate the gradient with respect to each of the parameters that we are interested in optimizing. The gradient is just the partial derivative with respect to each parameter. So for example, the gradient of the cost function ($J$) with respect to the weight that connects the third node in the input layer to the second node in the first hidden layer will be:

$$\dfrac{\partial{J}}{\partial{W^{[1]}_{23}}}$$

Each time we do backpropagation, we need to not only calculate this gradient, but the gradients for all of our parameters. How many gradients do we need to calculate? If we multiply the dimensions for each of our weights $(12 + 8 + 2 = 22)$ and biases $(4+2+1 = 7)$, we get 22 + 7 = 29 parameters and therefore 29 gradients (like the one above).

&nbsp;
#### Vectorizing the Gradients 

In the same way we combined the feature vectors $\textbf{x}$ with dimensions $(3,1)$ of $m$ training examples into a matrix $X$ with dimensions $(3, m)$, we can take our 29 gradients and combine them into gradient matricies to make our notation a little easier to follow. For example, we could represent the gradient of the cost function ($J$) with respect to $\textbf{W}^{[2]}$ as just a matrix of the partial derivatives of $J$ with respect to each entry $W^{[2]}_{ij}$ like so:

$$d\textbf{W}^{[2]} = 
\begin{bmatrix}
    \dfrac{\partial{J}}{\partial{W^{[2]}_{11}}} & 
    \dfrac{\partial{J}}{\partial{W^{[2]}_{12}}} &
    \dfrac{\partial{J}}{\partial{W^{[2]}_{13}}} &
    \dfrac{\partial{J}}{\partial{W^{[2]}_{14}}} \\\\
    \dfrac{\partial{J}}{\partial{W^{[2]}_{21}}} &
    \dfrac{\partial{J}}{\partial{W^{[2]}_{22}}} &
    \dfrac{\partial{J}}{\partial{W^{[2]}_{23}}} &
    \dfrac{\partial{J}}{\partial{W^{[2]}_{24}}} \\
\end{bmatrix}$$

Notice that the dimensions of the matrix $d\textbf{W}^{[2]}$ match $\textbf{W}^{[2]}$, which makes the gradient update a very simple elementwise operation.

So why isn't that matrix of partial derivatives equal to:

$$\dfrac{\partial{J}}{\partial{\textbf{W}^{[2]}}}$$

There are tons of resources online that treat these two things as the same thing. In fact, it wasn't until I took Andrew Ng's Deep Learning Specialization on Coursera that I was introduced to the notation $d\textbf{W}^{[2]}$. The problem is that for simple examples, they are equal to each other, but for more complex examples they won't be equal to each other. We will think of them as separate, and the partial derivative of $J$ with respect to  $\textbf{W}^{[2]}$ we can sometimes use as an intermediate calculation to arrive at $d\textbf{W}^{[2]}$.

We started with 29 gradients, and we can now collapse those gradients into 6 gradient matricies:

$$\bigg( d\textbf{W}^{[1]}, d\textbf{b}^{[1]}, d\textbf{W}^{[2]}, d\textbf{b}^{[2]}, 
d\textbf{W}^{[3]}, db^{[3]} \bigg)$$

Keep in mind that each of these gradient matricies should match the dimensions of the parameter they correspond to.

Next, let's talk about how we implement backpropagation to calculate these gradient matricies.

&nbsp;
## Implementing Backpropagation

In forward propagation, given a feature vector $\textbf{x}$ for the $ith$ example, our goal was to calculate one output, $\hat{y}$ which is our best guess for what class the example $i$ belongs to.

In backpropagation, for our 3 layer neural network example our goal is to calculate the 6 gradient matricies.

We do this (unsurprisingly) by working backwards. So we will start by calculating:

$$d\textbf{W}^{[3]}, db^{[3]}$$

What are the equations that connect our cost function $J$ to our weights and biases $\textbf{W}^{[3]}, b^{[3]}$ ? We are going to use $a^{[3]}$ instead of $\hat{y}$, but if it tickles your fancy you can feel free to use $\hat{y}$.

$$z^{[3]} = \textbf{W}^{[3]}a^{[2]} + b^{[3]}$$
$$a^{[3]} = \sigma(z^{[3]})$$
$$J(a^{[3]}, y) = -ylog(a^{[3]}) - (1-y)log(1 -a^{[3]})$$

So using the chain rule from calculus, we can think of $J$ as the composition of two other functions, $z^{[3]}$ and $a^{[3]}$ and thefore write the two gradient matricies as:

$$\dfrac{\partial{J}}{\partial{\textbf{W}^{[3]}}} = 
\dfrac{d{J}}{d{a^{[3]}}}
\dfrac{d{a^{[3]}}}{d{z^{[3]}}}
\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}}$$

$$\dfrac{\partial{J}}{\partial{b^{[3]}}} =
\dfrac{d{J}}{d{a^{[3]}}}
\dfrac{d{a^{[3]}}}{d{z^{[3]}}}
\dfrac{\partial{z^{[3]}}}{\partial{b^{[3]}}}$$

Before moving on, notice that the first equation calculates the partial derivative of $J$ (a scalar value) with respect to $\textbf{W}^{[3]}$, which is a row vector with dimensions $(1,2)$. So we are definitely going down the dark path of calculating vector and matrix derivatives. 

If you threw up in a little bit in your mouth at the prospect of taking a derivative with respect to a row vector, I found these [two](https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf) [resources](http://cs231n.stanford.edu/vecDerivs.pdf) helpful to understand the math a bit better.

Ok great, so now let's figure out what these 2 derivatives and 2 partial derivatives are equal to, so we can ultimately calculate the gradient of $J$ with respect to $\textbf{W}^{[3]}$ and $b^{[3]}$.

So why do some of the derivatives use $d$ and others use $\partial$? The simplest answer is that the ones that have a $d$ only depend on one variable, whereas the ones that use $\partial$ rely on more than one variable. So for example, when we think of the derivative of $\mathcal{L}$ with respect to $a^{[3]}$, we think of this equation:

$$J(a^{[3]} | y) = -ylog(a^{[3]}) - (1-y)log(1 -a^{[3]})$$

Which only uses one variable, $a^{[3]}$. Recall that $y$ is constant, since that is the label for the class that the training example $i$ belongs to. Since we only consider one variable, we use $d$.

But remember that $\mathcal{L}$ is a composition of two other functions, $z^{[3]}$ and $a^{[3]}$. So technically if we wrote everything out we would get:

$$J(\textbf{W}^{[3]}, b^{[3]} \ | \ y) = -ylog(\sigma(\textbf{W}^{[3]}a^{[2]} + b^{[3]})) - (1-y)log(1 - \sigma(\textbf{W}^{[3]}a^{[2]} + b^{[3]}))$$

Because we are now working with two variables, $\textbf{W}^{[3]}$ and $\textbf{W}^{[3]}$, we use $\partial$ to represent their derivatives instead of $d$.

I'm sure you can appreciate how I and I imagine a lot of other people get lost in the complexity of backpropagation. To calculate the first gradient matricies for a very simple network, we are already having to calculate 4 other gradients. My hope is that by going through this simple, 3-layer example you can scale to more complex models much easier than if you hand-waved your way through backpropagation. 

Let's start by calculating the derivative of $J$ with respect to $a^{[3]}$.

$$J(a^{[3]}, y) = -ylog(a^{[3]}) - (1-y)log(1 -a^{[3]})$$

We can take the derivative of both sides with respect to $a^{[3]}$ and end up with:

$$\dfrac{dJ}{d{a^{[3]}}} = \dfrac{-y}{a^{[3]}} - \dfrac{(1-y)}{1-a^{[3]}}(-1) $$
$$\dfrac{dJ}{d{a^{[3]}}} = \dfrac{-y}{a^{[3]}} + \dfrac{(1-y)}{1-a^{[3]}} $$

Let's do the same for the derivative of $a^{[3]}$ with respect to $z^{[3]}$:

$$a^{[3]} = \sigma(z^{[3]})$$

$$a^{[3]} = \dfrac{1}{1+e^{-z^{[3]}}}$$

$$a^{[3]} = (1+e^{-z^{[3]}})^{-1}$$

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = -(1+e^{-z^{[3]}})^{-2}(e^{-z^{[3]}})(-1)$$

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = \dfrac{e^{-z^{[3]}}}{(1+e^{-z^{[3]}})^2}$$

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = \dfrac{1}{1+e^{-z^{[3]}}}\dfrac{1}{1+e^{-z^{[3]}}}(e^{-z^{[3]}})$$

Note that:

$$a^{[3]} = \dfrac{1}{1+e^{-z^{[3]}}}$$
$$1+e^{-z^{[3]}} = \dfrac{1}{a^{[3]}}$$
$$e^{-z^{[3]}} = \dfrac{1-a^{[3]}}{a^{[3]}}$$

So back to the derivative, we can substitue things in and get:

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = (a^{[3]})^2\bigg(\dfrac{1-a^{[3]}}{a^{[3]}}\bigg)$$

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = a^{[3]}(1-a^{[3]})$$

Finally, we need to calculate the partial derivatives of $z^{[3]}$ with respect to $\textbf{W}^{[3]}$ and $b^{[3]}$. Up to this point, we've been calculating the deriviatives of scalars, and this is the first time we will calculate a gradient matrix. So the first gradient matrix we need to solve for is this guy:

$$\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}} = 
\begin{bmatrix}
    \dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{11}}} & \dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{12}}}
\end{bmatrix}$$

We call this matrix of partial derivatives a Jacobian matrix. What is a Jacobian Matrix? Let's spend a little bit of time deconstructing that.

&nbsp;
#### A simple Jacobian Matrix Explanation

Let's say we have a function $\mathcal{f}: \mathbb{R}^m \rightarrow \mathbb{R}^n$ that maps either a column or row vector with $m$ entries to one that has $n$ entries. Let's make the first vector a column vector called $\textbf{x}$ with $m$ entries.

$$f(\textbf{x}) = 
\begin{bmatrix}
    \\
    f_1(x_{11}, x_{21}, \dotsc, x_{m1}) \\\\
    f_2(x_{11}, x_{21}, \dotsc, x_{m1}) \\\\
    \vdots \\\\
    f_n((x_{11}, x_{21}, \dotsc, x_{m1})) \\\\
\end{bmatrix}$$

So $f(\textbf{x})$ is a (n, 1) column vector. It could also be a row vector, depending on how you define it.

So then, the Jacobian Matrix of $f$ with respect to $\textbf{x}$ is defined to be: 

$$ \dfrac{\partial\mathcal{f}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\\
\dfrac{\partial\mathcal{f_1}}{\partial x_{11}} &
\dotsc &
\dfrac{\partial\mathcal{f_1}}{\partial x_{m1}} \\\\
\vdots & \ddots & \vdots \\\\
\dfrac{\partial\mathcal{f_n}}{\partial x_{11}} &
\dotsc &
\dfrac{\partial\mathcal{f_n}}{\partial x_{m1}} \\\\
\end{bmatrix}
$$

Notice that the matrix has dimensions $(n, m)$. It gets its first dimension from the number of entries in the output vector $\mathcal{f}(\textbf{x})$ and its second dimensions from the number of entries in the input vector $\textbf{x}$, regardless of whether those vectors are column or row vectors. So actually, if $\textbf{x}$ was a row vector, we would get:

$$ \dfrac{\partial\mathcal{f}}{\partial\mathbf{x}} = 
\begin{bmatrix}
\\
\dfrac{\partial\mathcal{f_1}}{\partial x_{11}} &
\dotsc &
\dfrac{\partial\mathcal{f_1}}{\partial x_{1m}} \\\\
\vdots & \ddots & \vdots \\\\
\dfrac{\partial\mathcal{f_n}}{\partial x_{11}} &
\dotsc &
\dfrac{\partial\mathcal{f_n}}{\partial x_{1m}} \\\\
\end{bmatrix}
$$

See the slight difference? This kind of tripped me out when I first started thinking about it.

&nbsp;
#### Back to the Gradients

In order to solve for the two partial derivatives in the matrix, let's deconstruct our equation for $z^{[3]}$ so we use $W^{[3]}_{11}$ and $W^{[3]}_{12}$.

$$z^{[3]} = \textbf{W}^{[3]}a^{[2]} + b^{[3]}$$

$$z^{[3]} = 
\begin{bmatrix}
    W^{[3]}_{11} & W^{[3]}_{12}
\end{bmatrix}
\begin{bmatrix}
    a^{[2]}_{11} \\
    a^{[2]}_{21}
\end{bmatrix}+ b^{[3]}$$

$$z^{[3]} = W^{[3]}_{11}a^{[2]}_{11} + W^{[3]}_{12}a^{[2]}_{21}+ b^{[3]}$$

Now we can calculate the partial derivative of $z^{[3]}$ with respect to $W^{[3]}_{11}$ -

$$\dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{11}}} = (1)a^{[2]}_{11} + 0 + 0$$

$$\dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{11}}} = a^{[2]}_{11}$$

And the partial derivative of $z^{[3]}$ with respect to $W^{[3]}_{12}$ -

$$\dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{12}}} = (1)a^{[2]}_{21} + 0 + 0$$

$$\dfrac{\partial{z^{[3]}}}{\partial{W^{[3]}_{12}}} = a^{[2]}_{21}$$

So interestingly, the partial gradients with respect to the elements of the weight matrix $\textbf{W}^{[3]}$ are equal to the elements of the activations vector $a^{[2]}$ that they multiply with. This is a key point going forward when we try to generalize this process computing the larger gradient matricies.

Ok so now we will update our Jacobian matrix of $z^{[3]}$ with respect to $\textbf{W}^{[3]}$ and get the following:

$$\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}} = 
\begin{bmatrix}
    a^{[2]}_{11} & a^{[2]}_{21}
\end{bmatrix}$$

And just like a good Jacobian, the partial derivative has dimensions $(1,2)$, which match the number of entries in $z^{[3]}$ (1 since it's a scalar value) and the number of entries in $\textbf{W}^{[3]}$ (2).

Notice that $a^{[2]}$ has dimensions (2, 1), and the derivative has dimensions (1,2). On closer investigation, it's just the transpose of $a^{[2]}$. So another way to write it would be - 

$$\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}} = a^{[2]T}$$

Great! Let's move on to the calculating the partial derivative of $z^{[3]}$ with respect to $b^{[3]}$.

$$z^{[3]} = W^{[3]}_{11}a^{[2]}_{11} + W^{[3]}_{12}a^{[2]}_{21}+ b^{[3]}$$

$$\dfrac{\partial{z^{[3]}}}{\partial{b^{[3]}}} = 0 + 0 + 1$$

$$\dfrac{\partial{z^{[3]}}}{\partial{b^{[3]}}} = 1$$

And it's just 1! That's nice and simple. Now let's summarize all of our results from previous calculations and combine them to solve our original problem. So recall that our original problem was to calculate the partial derivative of the cost function $J$ with respect to the weights $W^{[3]}$ and biases $b^{[3]}$ connecting the second hidden layer $a^{[2]}$ to the output layer. We were able to deconstruct the derivative using the chain rule, and the results looked like this:

$$\dfrac{\partial{J}}{\partial{\textbf{W}^{[3]}}} = 
\dfrac{d{J}}{d{a^{[3]}}}
\dfrac{d{a^{[3]}}}{d{z^{[3]}}}
\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}}$$

$$\dfrac{\partial{\mathcal{L}}}{\partial{b^{[3]}}} =
\dfrac{d{\mathcal{L}}}{d{a^{[3]}}}
\dfrac{d{a^{[3]}}}{d{z^{[3]}}}
\dfrac{\partial{z^{[3]}}}{\partial{b^{[3]}}}$$

We spent the last few sections calculating the 4 intermediate derivatives needed to calculate the partial derivative of the loss function $\mathcal{L}$ with respect to the weights $\textbf{W}^{[3]}$ and biases $b^{[3]}$.

$$\dfrac{d{\mathcal{L}}}{d{a^{[3]}}} = \dfrac{-y}{a^{[3]}} + \dfrac{(1-y)}{1-a^{[3]}} $$

$$\dfrac{d{a^{[3]}}}{d{z^{[3]}}} = a^{[3]}(1-a^{[3]})$$

$$\dfrac{\partial{z^{[3]}}}{\partial{\textbf{W}^{[3]}}} = a^{[2]T}$$

$$\dfrac{\partial{z^{[3]}}}{\partial{b^{[3]}}} = 1$$


Let's simplify the derivative of the cost function $J$ with respect to $z^{[3]}$.
$$\dfrac{dJ}{d z^{[3]}} = \dfrac{dJ}{d{a^{[3]}}}\dfrac{d{a^{[3]}}}{d{z^{[3]}}}$$
$$ = \bigg(\dfrac{-y}{a^{[3]}} + \dfrac{(1-y)}{1-a^{[3]}}\bigg)a^{[3]}(1-a^{[3]})$$

$$= -y(1-a^{[3]}) + (1-y)a^{[3]}$$
$$= -y+ya^{[3]} + a^{[3]} -ya^{[3]}$$
$$=a^{[3]} - y$$

If we substitute these values in, we get:

$$\dfrac{\partial{J}}{\partial{\textbf{W}^{[3]}}} = (a^{[3]} - y) a^{[2]T}$$

$$\dfrac{\partial{J}}{\partial{b^{[3]}}} = a^{[3]} - y$$

And that's it! And luckily, the dimensions of our partial derivatives, $(1,2)$ and $1$ respectively, match the gradient matricies we wanted to calculate. So we get:

$$d\textbf{W}^{[3]} = (a^{[3]} - y) a^{[2]T}$$

$$d b^{[3]} = a^{[3]} - y$$

$$\bigg( d\textbf{W}^{[1]}, d\textbf{b}^{[1]}, d\textbf{W}^{[2]}, d\textbf{b}^{[2]}, 
d\textbf{W}^{[3]}, db^{[3]} \bigg)$$

We solved the first two gradient matricies $(d\textbf{W}^{[3]}, db^{[3]})$. Our goal was to solve for 29 gradients, and we just finished 3, since the partial derivative with respect to $\textbf{W}^{[3]}$ is a $(1, 2)$ matrix and therefore has two elements and the partial derivative with respect to $b^{[3]}$ has only one element.

Next, we continue to move backwards and focus on calculating the derivative of the loss function $\mathcal{L}$ with respect to $\textbf{W}^{[2]}$ and $\textbf{b}^{[2]}$. Starting with breaking it up using the chain rule, we get:

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[2]}}} = 
\dfrac{d{J}}{d{z^{[3]}}}
\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}}
\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}}
\dfrac{\partial{\textbf{z}^{[2]}}}{\partial{\textbf{W}^{[2]}}}
$$

$$
\dfrac{\partial{J}}{\partial{\textbf{b}^{[2]}}} = 
\dfrac{d{\mathcal{L}}}{d{z^{[3]}}}
\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}}
\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}}
\dfrac{\partial{\textbf{z}^{[2]}}}{\partial{\textbf{b}^{[2]}}}
$$

Notice that since:

$$
\dfrac{d{J}}{d{z^{[3]}}} = 
\dfrac{d{J}}{d{a^{[3]}}}
\dfrac{d{a^{[3]}}}{d{z^{[3]}}}
$$

We already know that it is equal to:

$$
\dfrac{d{J}}{d{z^{[3]}}} = a^{[3]} - y$$

And don't need to calculate it again.

We sometimes refer to the partial derivative of the cost function $J$ with respect to $z^{[3]}$ as $\delta^{[3]}$, and will use that notation for the rest of this blog post.

Let's focus on what we need to calculate. Let's start with the derivative of $\textbf{z}^{[3]}$ with respect to $\textbf{z}^{[2]}$, which we can break down into two steps:

$$
\dfrac{d{z^{[3]}}}{d{\textbf{z}^{[2]}}} = 
\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}}
\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}}
$$


We will start with calculating the derivative of $\textbf{z}^{[3]}$ with respect to the activation $\textbf{a}^{[2]}$.

Recall that $z^{[3]}$ matches the number of nodes in the output layer and is therefore a scalar value. $\textbf{a}^{[2]}$ is a $(2,1)$ column vector and its number of entries match the number of nodes in the 3rd layer or the 2nd hidden layer. So the Jacobian matrix looks like:

$$\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}} = 
\begin{bmatrix}
    \dfrac{\partial{z^{[3]}}}{\partial{a^{[2]}_{11}}} &
    \dfrac{\partial{z^{[3]}}}{\partial{a^{[2]}_{21}}} 
\end{bmatrix}$$

And the equation involving $z^{[3]}$ and $\textbf{a}^{[2]}$ is:

$$z^{[3]} = \textbf{W}^{[3]}\textbf{a}^{[2]} + b^{[3]}$$

Let's break this equation down:

$$z^{[3]} = 
W^{[3]}_{11}a^{[2]}_{11} + 
W^{[3]}_{12}a^{[2]}_{21} +
b^{[3]}
$$

If we take the partial derivative of $z^{[3]}$ with respect to $a^{[2]}_{21}$, we get:

$$\dfrac{\partial{z^{[3]}}}{\partial{a^{[2]}_{21}}} =
0 + 
W^{[3]}_{12}(1) +
0
$$

$$\dfrac{\partial{z^{[3]}}}{\partial{a^{[2]}_{21}}} =
W^{[3]}_{12}
$$

Applying this logic to every partial derivative in the vector, we get:

$$\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}} = 
\begin{bmatrix}
\\
W^{[3]}_{11} & 
W^{[3]}_{12}
\\\\
\end{bmatrix}$$

Next, we need to calculate the derivative of $\textbf{a}^{[2]}$ with respect to $\textbf{z}^{[2]}$. 

Both $\textbf{a}^{[2]}$ and $\textbf{z}^{[2]}$ are column vectors with dimensions $(2,1)$ and therefore the Jacobian matrix of $\textbf{a}^{[2]}$ with respect to $\textbf{z}^{[2]}$ will be: 

$$\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}} = 
\begin{bmatrix}
\\
    \dfrac{\partial{a^{[3]}_{11}}}{\partial{z^{[2]}_{11}}} &
    \dfrac{\partial{a^{[3]}_{11}}}{\partial{z^{[2]}_{21}}} \\\\
    \dfrac{\partial{a^{[3]}_{21}}}{\partial{z^{[2]}_{11}}} &
    \dfrac{\partial{a^{[3]}_{21}}}{\partial{z^{[2]}_{21}}} \\\\
\end{bmatrix}$$

Notice that in order to go from $\textbf{z}^{[2]} \rightarrow \textbf{a}^{[2]}$, we apply the ReLU function $g(z)$ to each element in $\textbf{z}^{[2]}$. So for example, to calculate $a^{[2]}_{21}$:

$$a^{[2]}_{21} = g(z^{[2]}_{21})$$

Applying this logic to every partial derivative in the vector, we get:

$$\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}} = 
\begin{bmatrix}
\\
    g'(z^{[2]}_{11}) & 0 \\\\ 
    0 & g'(z^{[2]}_{21})
\\\\
\end{bmatrix}$$

Where $g'(z)$ is the derivative of the ReLU function $g(z)$.

What is the derivative of ReLU equal to?

$$
g'(z) = \begin{cases}
   1 &\text{if } z > 0  \\
   \text{Undefined} &\text{if } z = 0  \\
   0 &\text{if } z < 0
\end{cases}
$$

Why is the derivative of $g$ undefined when $z = 0$? For a function  to be differentiable at a point, it has to be continuous at that point. In order for a point to be continuous at a point, the limit of the function as it approaches that point has to defined. In order for the limit to be defined, the left and right hand limits have to equal. In this case, the right hand limit is equal to 1 and the left hand limit is equal to 0. Therefore, ReLU is not differentiable at 0.

Does it matter that the derivative of the ReLU function is undefined at $z=0$ for backpropagation? In practice, no since $z$ will never be truly equal to $0$ - software implementations will have a rounding error for float points.

Finally, we will need to calculate the partial derivatives of $\textbf{z}^{[2]}$ with respect to $\textbf{W}^{[2]}$ and $\textbf{b}^{[2]}$.

Let's first focus on the partial derivative of $\textbf{z}^{[2]}$ with respect to $\textbf{W}^{[2]}$. $\textbf{z}^{[2]}$ is a one-dimensional array $(2,1)$ and $\textbf{W}^{[2]}$ is a two-dimensional matrix $(2,4)$. Notice that each entry in $\textbf{z}^{[2]}$ can be though of as the result of a function involving all 8 of the weight entries in $\textbf{W}^{[2]}$. So for example, $z^{[2]}_{11}$ can be though of as:

$$z^{[2]}_{11} = f_1(W^{[2]}_{11}, \dotsc, W^{[2]}_{14},W^{[2]}_{21},\dotsc W^{[2]}_{24} | \textbf{a}^{[2]}, \textbf{b}^{[2]})$$

$$z^{[2]}_{11} = W^{[2]}_{11}a^{[2]}_{11} + \dotsc + W^{[2]}_{14}a^{[2]}_{41} +  b^{[2]}_{11}$$

If we represented the entries in $\textbf{z}^{[2]}$ in this way, we get:

$$\textbf{z}^{[2]} = 
\begin{bmatrix}
f_1(W^{[2]}_{11}, W^{[2]}_{12}, W^{[2]}_{13},W^{[2]}_{14},W^{[2]}_{21},\dotsc W^{[2]}_{24} | \textbf{a}^{[2]}, \textbf{b}^{[2]}) \\\\
f_1(W^{[2]}_{11}, W^{[2]}_{12}, W^{[2]}_{13},W^{[2]}_{14},W^{[2]}_{21},\dotsc W^{[2]}_{24} | \textbf{a}^{[2]}, \textbf{b}^{[2]})
\end{bmatrix}
$$

And the Jacobian matrix would look like:

$$
\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{W}^{[2]}} = 
\begin{bmatrix}
\dfrac{\partial z^{[2]}_{11}}{\partial W^{[2]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[2]}_{11}}{\partial W^{[2]}_{14}} &
\dfrac{\partial z^{[2]}_{11}}{\partial W^{[2]}_{21}} &
\dotsc &
\dfrac{\partial z^{[2]}_{11}}{\partial W^{[2]}_{24}} \\\\
\dfrac{\partial z^{[2]}_{21}}{\partial W^{[2]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[2]}_{21}}{\partial W^{[2]}_{14}} &
\dfrac{\partial z^{[2]}_{21}}{\partial W^{[2]}_{21}} &
\dotsc &
\dfrac{\partial z^{[2]}_{21}}{\partial W^{[2]}_{24}} \\\\
\end{bmatrix}
$$

So the Jacobian matrix has dimensions $(2, 8)$, which is equal to the number of entries in $\textbf{z}^{[2]}$ and the number of entries in $\textbf{W}^{[2]}$.

But do we need to calculate all 16 derivatives? Luckily, no. The reason is that most of the derivatives will be equal to $0$, and will stay $0$ regardless if we change the values of any of the variables.

In order to illustrate this point, let's look at the first element of $z^{[2]}$:

$$z^{[2]}_{11} = W^{[2]}_{11}a^{[1]}_{11} + W^{[2]}_{12}a^{[1]}_{21} + W^{[2]}_{13}a^{[1]}_{31} + W^{[2]}_{14}a^{[1]}_{41} $$

Notice that in order to calculate the partial derivative of $z^{[2]}_1$ with respect to $W^{[2]}$, we only need to worry about $W^{[2]}_{11}$, $W^{[2]}_{12}$, $W^{[2]}_{13}$, and $W^{[2]}_{14}$ and not the other 4 scalar variables in $W^{[2]}$. As a general rule of thumb, the partial derivative of a scalar element in vector $a$ with respect to a scalar element in matrix $W$ will be nonzero when the x-dimension of $W$ matches the dimension of $a$.

And from a previous calculation, we know that the derivative of a scalar component of $z$ with respect to a scalar component of $W$ is just that scalar component of $W$ if it's included in the calculation of the scalar component of $z$, so we get this fun matrix:

$$
\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{W}^{[2]}} = 
\begin{bmatrix}
a^{[2]}_{11} &
\dotsc & 
a^{[2]}_{41} &
0 &
\dotsc &
0 \\\\
0 &
\dotsc & 
0 &
a^{[2]}_{11} & 
\dotsc &
a^{[2]}_{41} & \\\\
\end{bmatrix}
$$

Now, let's calculate the Jacobian matrix of $\textbf{z}^{[2]}$ with respect to $\textbf{b}^{[2]}$. It's pretty simple, just like last time the derivatives that involve the entry of $\textbf{b}^{[2]}$ are equal to 1.

$$\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{b}^{[2]}} = 
\begin{bmatrix}
    1 & 0 \\\\
    0 & 1 
\end{bmatrix}$$

The dimensions $(2,2)$ of the Jacobian matrix are equal to the number of entries in $\textbf{z}^{[2]}$ and $\textbf{b}^{[2]}$, respectively.

We have all the pieces to finally calculate the derivative of the cost function $J$ with respect to $\mathbf{W}^{[2]}$ and $\mathbf{b}^{[2]}$. 

$$
\dfrac{\partial{J}}{\partial{\mathbf{W}^{[2]}}} = 
\delta^{[3]}
\dfrac{d{z^{[3]}}}{d{\mathbf{z}^{[2]}}}
\dfrac{\partial{\mathbf{z}^{[2]}}}{\partial{\mathbf{W}^{[2]}}}
$$

And this is how we deconstructed the derivative of the loss function $\mathcal{L}$ with respect to $b^{[2]}$.

$$
\dfrac{\partial{J}}{\partial{\mathbf{b}^{[2]}}} = 
\delta^{[3]}
\dfrac{d{z^{[3]}}}{d{\mathbf{z}^{[2]}}}
\dfrac{\partial{\mathbf{z}^{[2]}}}{\partial{\mathbf{b}^{[2]}}}
$$

To summarize our calculations and the dimensions of each:

$$
\delta^{[3]} = 
a^{[3]} - y
$$
Is a scalar value.

$$\dfrac{d{z^{[3]}}}{d{\textbf{a}^{[2]}}} = 
\begin{bmatrix}
\\
    W^{[3]}_{11} & 
    W^{[3]}_{12}
\\\\
\end{bmatrix}$$

Is a $(1,2)$ matrix.

$$\dfrac{d{\textbf{a}^{[2]}}}{d{\textbf{z}^{[2]}}} = 
\begin{bmatrix}
\\
    g'(z^{[2]}_{11}) & 0 \\\\ 
    0 & g'(z^{[2]}_{21})
\\\\
\end{bmatrix}$$

Is a $(2, 2)$ matrix.

$$
\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{W}^{[2]}} = 
\begin{bmatrix}
a^{[2]}_{11} &
\dotsc & 
a^{[2]}_{41} &
0 &
\dotsc &
0 \\\\
0 &
\dotsc & 
0 &
a^{[2]}_{11} & 
\dotsc &
a^{[2]}_{41} & \\\\
\end{bmatrix}
$$

Is a $(2,8)$ matrix.

$$\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{b}^{[2]}} = 
\begin{bmatrix}
    1 & 0 \\\\
    0 & 1 
\end{bmatrix}$$

Is a $(2,2)$ matrix.

Let's first try and calculate the derivative of $J$ with respect to $\textbf{W}^{[2]}$. We can start by looking at:

$$\dfrac{\partial{J}}{\partial{\mathbf{z}^{[2]}}} = 
\delta^{[3]}
\dfrac{d{z^{[3]}}}{d{\mathbf{a}^{[2]}}}
\dfrac{d{a^{[3]}}}{d{\mathbf{z}^{[2]}}}$$

$$\dfrac{\partial{J}}{\partial{\mathbf{z}^{[2]}}} = \delta^{[3]}
\begin{bmatrix}
W^{[3]}_{11} & 
W^{[3]}_{12}
\end{bmatrix}
\begin{bmatrix}
g'(z^{[2]}_{11}) & 0 \\\\ 
0 & g'(z^{[2]}_{21})
\end{bmatrix}
$$

So this is the Jacobian Matrix of $J$ with respect to $\textbf{z}^{[2]}$. But notice that this matrix is $(1,2)$ but $\textbf{z}^{[2]}$ is $(2,1)$. We can define a gradient matrix similar to what we did for the weights and bias parameters that match the dimensions of $\textbf{z}^{[2]}$ called $\boldsymbol{\delta}^{[2]}$:

$$
\boldsymbol{\delta}^{[2]} = 
\bigg( \dfrac{\partial{J}}{\partial{\mathbf{z}^{[2]}}}\bigg)^T
$$

Recall that given three matricies $\textbf{A}$, $\textbf{B}$, and $\textbf{d}$ the $(\textbf{ABC})^T = \textbf{C}^T\textbf{B}^T\textbf{A}^T$

$$
\boldsymbol{\delta}^{[2]} = 
\begin{bmatrix}
g'(z^{[2]}_{11}) & 0 \\\\ 
0 & g'(z^{[2]}_{21})
\end{bmatrix}
\begin{bmatrix}
W^{[3]}_{11} \\\\ 
W^{[3]}_{12}
\end{bmatrix}
\delta^{[3]}
$$

$$
\boldsymbol{\delta}^{[2]} = 
\begin{bmatrix}
g'(z^{[2]}_{11})W^{[3]}_{11}\delta^{[3]} \\\\ 
g'(z^{[2]}_{21})W^{[3]}_{12}\delta^{[3]}
\end{bmatrix}
$$

Notice when we rearranged the products in the matrix, we get:

$$
\boldsymbol{\delta}^{[2]} = 
\begin{bmatrix}
W^{[3]}_{11}\delta^{[3]}g'(z^{[2]}_{11}) \\\\ 
W^{[3]}_{12}\delta^{[3]}g'(z^{[2]}_{21})
\end{bmatrix}
$$

$$
\boldsymbol{\delta}^{[2]} = 
\begin{bmatrix}
W^{[3]}_{11}\delta^{[3]} \\\\ 
W^{[3]}_{12}\delta^{[3]}
\end{bmatrix}
*
g'(\textbf{z}^{[2]})
$$

Where $*$ indicates elementwise multiplication between two matricies. $g'(\textbf{z}^{[2]})$ is a columnwise vector of the derivative of ReLU $g'(z)$ applied to each entry of $\textbf{z}^{[2]}$.

Next, we can decompose the result into:

$$
\boldsymbol{\delta}^{[2]}= 
\begin{bmatrix}
W^{[3]}_{11} \\\\ 
W^{[3]}_{12}
\end{bmatrix}
\delta^{[3]}
*
g'(\textbf{z}^{[2]})
$$

$$
\boldsymbol{\delta}^{[2]}= 
\textbf{W}^{[3]T}
\delta^{[3]}
*
g'(\textbf{z}^{[2]})
$$

This final result the gradient of the cost function $J$ with respect to $\textbf{z}^{[2]}$. It is a $(2,1)$ column vector with dimensions that match $\textbf{z}^{[2]}$.

Plugging that result into our equation we get:

$$
\dfrac{\partial{J}}{\partial{\mathbf{W}^{[2]}}} = 
\dfrac{\partial{J}}{\partial{\mathbf{z}^{[2]}}}
\dfrac{\partial{\mathbf{z}^{[2]}}}{\partial{\mathbf{W}^{[2]}}}
$$

$$
\dfrac{\partial{J}}{\partial{\mathbf{W}^{[2]}}} = 
\boldsymbol{\delta}^{[2]T}
\begin{bmatrix}
a^{[2]}_{11} &
\dotsc & 
a^{[2]}_{41} &
0 &
\dotsc &
0 \\\\
0 &
\dotsc & 
0 &
a^{[2]}_{11} & 
\dotsc &
a^{[2]}_{41} & \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}
\delta^{[2]}_{11} &
\delta^{[2]}_{21}
\end{bmatrix}
\begin{bmatrix}
a^{[2]}_{11} &
\dotsc & 
a^{[2]}_{41} &
0 &
\dotsc &
0 \\\\
0 &
\dotsc & 
0 &
a^{[2]}_{11} & 
\dotsc &
a^{[2]}_{41} & \\\\
\end{bmatrix}
$$

And we get this fun $(1, 8)$ Jacobian matrix:

$$\dfrac{\partial{J}}{\partial{\textbf{W}^{[2]}}} = 
\begin{bmatrix}
\delta^{[2]}_{11}a^{[2]}_{11} & 
\delta^{[2]}_{11}a^{[2]}_{21} &
\delta^{[2]}_{11}a^{[2]}_{31} & 
\delta^{[2]}_{11}a^{[2]}_{41} &
\delta^{[2]}_{21}a^{[2]}_{11} & 
\delta^{[2]}_{21}a^{[2]}_{21} &
\delta^{[2]}_{21}a^{[2]}_{31} & 
\delta^{[2]}_{21}a^{[2]}_{41}
\end{bmatrix}
$$

So this is our Jacobian. But we need our gradient matrix $d\textbf{W}^{[2]}$ to have dimensions that match $\textbf{W}^{[2]}$. So we will reshape the Jacobian into a $(2,4)$ matrix.

$$d\textbf{W}^{[2]} = 
\begin{bmatrix}
\delta^{[2]}_{11}a^{[2]}_{11} & 
\dotsc &
\delta^{[2]}_{11}a^{[2]}_{41} \\\\
\delta^{[2]}_{21}a^{[2]}_{11} & 
\dotsc & 
\delta^{[2]}_{21}a^{[2]}_{41} \\\\
\end{bmatrix}
$$

And, interestingly, we can break this apart into two matricies.

$$d\textbf{W}^{[2]} = 
\begin{bmatrix}
\delta^{[2]}_{11} \\\\
\delta^{[2]}_{21} \\\\
\end{bmatrix}
\begin{bmatrix}
a^{[2]}_{11} & a^{[2]}_{21} &  a^{[2]}_{31} & a^{[2]}_{41} \\\\
\end{bmatrix}
$$

Which becomes:

$$d\textbf{W}^{[2]} = \boldsymbol{\delta}^{[2]}\textbf{a}^{[2]T}$$

Alright, so you might be wondering, why did we need to go through all that work to arrive at that simple result? The reason is because most tutorials, blog posts, and courses skip the math and arrive at this result. But I think it's important to work through how we arrive there step by step. When we go through it step by step, we begin to understand how each of these operations relates to linear algebra and multivariate calculus. When we are just presented with the final result, we tend to just memorize it.

Let's now calculate the partial derivative of the loss $J$ with respect to $\textbf{b}^{[2]}$, which luckily is a lot easier.

$$\dfrac{\partial J}{\partial{\textbf{b}^{[2]}}} = \dfrac{dJ}{d{\textbf{z}^{[2]}}}
\dfrac{\partial{\textbf{z}^{[2]}}}{\partial{\textbf{b}^{[2]}}}$$

We know that:

$$\dfrac{\partial \textbf{z}^{[2]}}{\partial \textbf{b}^{[2]}} = 
\begin{bmatrix}
    1 & 0 \\\\
    0 & 1 
\end{bmatrix}$$

Which is just the identity matrix. So the partial derivative just simplfies to become:

$$\dfrac{\partial{J}}{\partial{\textbf{b}^{[2]}}} = \boldsymbol{\delta}^{[2]T}$$


And the gradient $d\textbf{b}^{[2]}$ is therefore just the transpose of the Jacobian, or:

$$
d\textbf{b}^{[2]} = \boldsymbol{\delta}^{[2]}
$$

Which is a $(2,1)$ column vector, and matches the dimensions of $\textbf{b}^{[2]}$.

Great! So we calculated two more gradient matricies (four in total), and we still have two more to go.

$$\bigg( d\textbf{W}^{[1]}, d\textbf{b}^{[1]}, d\textbf{W}^{[2]}, d\textbf{b}^{[2]}, 
d\textbf{W}^{[3]}, db^{[3]} \bigg)$$

We also said that by breaking down all the gradient matricies into their respective partial derivatives, we needed to solve for 29. We solved for $3$ in layer 3, and $8 + 2 = 10$ in the second layer. 16 more to go!

&nbsp;
### Calculating $d\textbf{W}^{[1]}$ and $d\textbf{b}^{[1]}$

So by this point you should know the drill. In order to calculate our gradient matricies, we need to calculate the Jacobian Matricies of $J$ with respect to $\textbf{W}^{[1]}$ and $\textbf{b}^{[1]}$ 

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[1]}}} = 
\dfrac{d{J}}{d{\textbf{z}^{[2]}}}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{a}^{[1]}}}
\dfrac{d\textbf{a}^{[1]}}{d{\textbf{z}^{[1]}}}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{W}^{[1]}}}
$$

$$
\dfrac{\partial{J}}{\partial{\textbf{b}^{[1]}}} = 
\dfrac{d{J}}{d{\textbf{z}^{[2]}}}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{a}^{[1]}}}
\dfrac{d{\textbf{a}^{[1]}}}{d{\textbf{a}^{[1]}}}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{b}^{[1]}}}
$$

We already know that:

$$
\dfrac{d{J}}{d{\mathbf{z}^{[2]}}} = \delta^{[2]T}$$

Which is a $(1,2)$ matrix.

We are going to lump a couple of derivatives together.

$$
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}} =
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{a}^{[1]}}}
\dfrac{d\textbf{a}^{[1]}}{d{\textbf{z}^{[1]}}}
$$

And just solve for the Jacobian matrix of $\textbf{z}^{[2]}$ with respect to $\textbf{z}^{[1]}$. 

$$\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}} = 
\begin{bmatrix}
\\
    \dfrac{\partial{z^{[2]}_{11}}}{\partial{z^{[1]}_{11}}} & 
    \dfrac{\partial{z^{[2]}_{11}}}{\partial{z^{[1]}_{21}}} &
    \dfrac{\partial{z^{[2]}_{11}}}{\partial{z^{[1]}_{31}}} &
    \dfrac{\partial{z^{[2]}_{11}}}{\partial{z^{[1]}_{41}}} \\\\
    \dfrac{\partial{z^{[2]}_{21}}}{\partial{z^{[1]}_{11}}} & 
    \dfrac{\partial{z^{[2]}_{21}}}{\partial{z^{[1]}_{21}}} &
    \dfrac{\partial{z^{[2]}_{21}}}{\partial{z^{[1]}_{31}}} &
    \dfrac{\partial{z^{[2]}_{21}}}{\partial{z^{[1]}_{41}}} \\\\
\end{bmatrix}$$

With dimensions $(2,4)$. Again, the first dimension matches the number of entries in $\textbf{z}^{[2]}$ and the second dimension mathces the number of entries in $\textbf{z}^{[1]}$  Similar to the previous layer, when we plug in values we get this matrix:

$$
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}} = 
\begin{bmatrix}\\
    W^{[2]}_{11} & 
    \dotsc &
    W^{[2]}_{14}\\\\
    W^{[2]}_{21} & 
    \dotsc &
    W^{[2]}_{24} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    g'(z^{[1]}_{11}) & 0 & 0 & 0 \\\\
    0 & g'(z^{[1]}_{21}) & 0 & 0 \\\\
    0 & 0 & g'(z^{[1]}_{31}) & 0 \\\\
    0 & 0 & 0 &g'(z^{[1]}_{41}) \\\\
\end{bmatrix}
$$

Alright, now we need to calculate the partial derivative of $\textbf{z}^{[1]}$ with respect to $\textbf{W}^{[1]}$. Like last time, we can construct the Jacobian matrix:

$$
\dfrac{\partial \textbf{z}^{[1]}}{\partial \textbf{W}^{[1]}} = 
\begin{bmatrix}
\dfrac{\partial z^{[1]}_{11}}{\partial W^{[1]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[1]}_{11}}{\partial W^{[1]}_{43}} \\\\
\dfrac{\partial z^{[1]}_{21}}{\partial W^{[1]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[1]}_{21}}{\partial W^{[1]}_{43}} \\\\
\dfrac{\partial z^{[1]}_{31}}{\partial W^{[1]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[1]}_{31}}{\partial W^{[1]}_{43}} \\\\
\dfrac{\partial z^{[1]}_{41}}{\partial W^{[1]}_{11}} &
\dotsc & 
\dfrac{\partial z^{[1]}_{41}}{\partial W^{[1]}_{43}} \\\\
\end{bmatrix}
$$


So the Jacobian matrix has dimensions $(4, 12)$, which is equal to the number of entries in $\textbf{z}^{[1]}$ and the number of entries in $\textbf{W}^{[1]}$ $(4*3 = 12)$.

And similar to our previous calculation, it simplifies to become:

$$
\dfrac{\partial \textbf{z}^{[1]}}{\partial \textbf{W}^{[1]}} = 
\begin{bmatrix}
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 &
0 &
0 &
0 &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
0 &
0 &
0 &
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
0 &
0 &
0 &
0 &
0 &
0 &
x_{11} &
x_{21} & 
x_{31} & \\\\
\end{bmatrix}
$$

Like last time, the Jacobian matrix of $\textbf{z}^{[1]}$ with respect to $\textbf{b}^{[1]}$ is equal to the identity matrix:

$$\dfrac{\partial \textbf{z}^{[1]}}{\partial \textbf{b}^{[1]}} = 
\begin{bmatrix}
\\
    1 & 0 & 0 & 0 \\\\
    0 & 1 & 0 & 0 \\\\
    0 & 0 & 1 & 0 \\\\
    0 & 0 & 0 & 1 \\\\
\end{bmatrix}$$

The dimensions $(4,4)$ of the Jacobian matrix are equal to the number of entries in $\textbf{z}^{[1]}$ and $\textbf{b}^{[1]}$, respectively.

Now we have the intermediate pieces to calculate the partial derivative of $J$ with respect to $\textbf{W}^{[1]}$ and $\textbf{b}^{[1]}$, which will eventually allow us to calculate the gradient matrix.

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[1]}}} = 
\dfrac{d{J}}{d{\textbf{z}^{[2]}}}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{W}^{[1]}}}
$$

###

$$
\dfrac{\partial{J}}{\partial{\textbf{b}^{[1]}}} = 
\dfrac{d{J}}{d{\textbf{z}^{[2]}}}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{b}^{[1]}}}
$$

Let's start by calculating the derivative of $J$ with respect to $\textbf{z}^{[1]}$:

$$
\dfrac{d{J}}{d{\textbf{z}^{[1]}}} = 
\dfrac{d{J}}{d{\textbf{z}^{[2]}}}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}}
$$

$$
\dfrac{d{J}}{d{\mathbf{z}^{[2]}}} = 
\boldsymbol{\delta}^{[2]T}
\dfrac{d{\textbf{z}^{[2]}}}{d{\textbf{z}^{[1]}}}$$

So this is the Jacobian Matrix of $J$ with respect to $\textbf{z}^{[2]}$. Let's figure out the dimensions for this matrix based on the dimensions of its components. $\boldsymbol{\delta}^{[2]T}$ has dimensions (1, 2), and the derivative of $\textbf{z}^{[2]}$ with respect to $\textbf{z}^{[1]}$ has dimensions $(2,4)$. Doing the matrix multiply, we get a matrix of dimensions $(1,4)$, which matches what we would expect. 

$$
\dfrac{dJ}{d{\textbf{z}^{[1]}}} = 
\boldsymbol{\delta}^{[2]T}
\begin{bmatrix}\\
    W^{[2]}_{11} & 
    \dotsc &
    W^{[2]}_{14} \\\\
    W^{[2]}_{21} & 
    \dotsc &
    W^{[2]}_{24} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    g'(z^{[1]}_{11}) & 0 & 0 & 0 \\\\
    0 & g'(z^{[1]}_{21}) & 0 & 0 \\\\
    0 & 0 & g'(z^{[1]}_{31}) & 0 \\\\
    0 & 0 & 0 &g'(z^{[1]}_{41}) \\\\
\end{bmatrix}
$$

To calculate the gradient of the cost function $J$ with respect to $\textbf{z}^{[1]}$, we need to take the transpose of the Jacobian matrix.

$$
\boldsymbol{\delta}^{[1]} = 
\begin{bmatrix}\\
    g'(z^{[1]}_{11}) & 0 & 0 & 0 \\\\
    0 & g'(z^{[1]}_{21}) & 0 & 0 \\\\
    0 & 0 & g'(z^{[1]}_{31}) & 0 \\\\
    0 & 0 & 0 &g'(z^{[1]}_{41}) \\\\
\end{bmatrix}
\begin{bmatrix}\\
    W^{[2]}_{11} & 
    W^{[2]}_{21} \\\\
    \vdots & \vdots \\\\
    W^{[2]}_{14} & 
    W^{[2]}_{24} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    \delta^{[1]}_{11} \\\\
    \delta^{[1]}_{21} \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}\\
    g'(z^{[1]}_{11})W^{[2]}_{11} & 
    g'(z^{[1]}_{11})W^{[2]}_{21} \\\\
    \vdots & \vdots \\\\
    g'(z^{[1]}_{41})W^{[2]}_{14} & 
    g'(z^{[1]}_{41})W^{[2]}_{24} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    \delta^{[1]}_{11} \\\\
    \delta^{[1]}_{21} \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}\\
    g'(z^{[1]}_{11})W^{[2]}_{11}\delta^{[1]}_{11} + 
    g'(z^{[1]}_{11})W^{[2]}_{21}\delta^{[1]}_{21} \\\\
    \vdots & \\\\
    g'(z^{[1]}_{41})W^{[2]}_{14}\delta^{[1]}_{11} + 
    g'(z^{[1]}_{41})W^{[2]}_{24}\delta^{[1]}_{21} \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}\\
    g'(z^{[1]}_{11})(W^{[2]}_{11}\delta^{[1]}_{11} + 
    W^{[2]}_{21}\delta^{[1]}_{21}) \\\\
    \vdots & \\\\
    g'(z^{[1]}_{41})(W^{[2]}_{14}\delta^{[1]}_{11} + 
    W^{[2]}_{24}\delta^{[1]}_{21}) \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}\\
    W^{[2]}_{11}\delta^{[1]}_{11} + 
    W^{[2]}_{21}\delta^{[1]}_{12} \\\\
    \vdots & \\\\
    W^{[2]}_{14}\delta^{[1]}_{11} + 
    W^{[2]}_{24}\delta^{[1]}_{12} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    g'(z^{[1]}_{11}) \\\\
    \vdots & \\\\
    g'(z^{[1]}_{41}) \\\\
\end{bmatrix}
$$

$$
 = 
\begin{bmatrix}\\
    W^{[2]}_{11}\delta^{[1]}_{11} + 
    W^{[2]}_{21}\delta^{[1]}_{12} \\\\
    \vdots & \\\\
    W^{[2]}_{14}\delta^{[1]}_{11} + 
    W^{[2]}_{24}\delta^{[1]}_{12} \\\\
\end{bmatrix}
* g'(\textbf{z}^{[1]})
$$

$$
 = 
\begin{bmatrix}\\
    W^{[2]}_{11} & 
    W^{[2]}_{21} \\\\
    \vdots & \vdots \\\\
    W^{[2]}_{14} & 
    W^{[2]}_{24} \\\\
\end{bmatrix}
\begin{bmatrix}\\
    \delta^{[1]}_{11} \\\\
    \delta^{[1]}_{12} \\\\
\end{bmatrix}
* g'(\textbf{z}^{[1]})
$$

$$
\boldsymbol{\delta}^{[1]} = 
\textbf{W}^{[2]T}\boldsymbol{\delta}^{[1]}
* g'(\textbf{z}^{[1]})
$$

Great, now we can use $\boldsymbol{\delta}^{[1]}$ to calculate the derivative of $J$ with respect to $\textbf{W}^{[1]}$.

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[1]}}} = 
\boldsymbol{\delta}^{[1]T}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{W}^{[1]}}}
$$

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[1]}}} = 
\begin{bmatrix}
\delta^{[1]}_{11} & \delta^{[1]}_{21} & \delta^{[1]}_{31} & \delta^{[1]}_{41}
\end{bmatrix}
\begin{bmatrix}
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 &
0 &
0 &
0 &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
0 &
0 &
0 &
x_{11} &
x_{21} & 
x_{31} &
0 &
0 &
0 \\\\
0 &
0 & 
0 &
0 &
0 &
0 &
0 &
0 &
0 &
x_{11} &
x_{21} & 
x_{31} & \\\\
\end{bmatrix}
$$

$$
\dfrac{\partial{J}}{\partial{\textbf{W}^{[1]}}} =
\begin{bmatrix}
\delta^{[1]}_{11}x_{11} & 
\delta^{[1]}_{11}x_{21} &
\delta^{[1]}_{11}x_{31} &
\dotsc
\delta^{[1]}_{41}x_{11} & 
\delta^{[1]}_{41}x_{21} &
\delta^{[1]}_{41}x_{31} 
\end{bmatrix}
$$

Which yields a $(1, 12)$ Jacobian matrix. Like before, we need our gradient matrix $d\textbf{W}^{[1]}$ to have dimensions that match $\textbf{W}^{[1]}$. So we will reshape the Jacobian into a $(4,3)$ matrix.

$$d\textbf{W}^{[1]} = 
\begin{bmatrix}
\\
\delta^{[1]}_{11}x_{11} & 
\dotsc &
\delta^{[1]}_{11}x_{31} \\\\
\vdots & \vdots \\\\
\delta^{[1]}_{41}x_{11} & 
\dotsc & 
\delta^{[1]}_{41}x_{31} \\\\
\end{bmatrix}
$$

And like last time, this breaks apart into two matricies and we get:

$$d\textbf{W}^{[1]} = 
\begin{bmatrix}
\\
\delta^{[1]}_{11} \\\\
\delta^{[1]}_{21} \\\\
\delta^{[1]}_{31} \\\\
\delta^{[1]}_{41} \\\\
\end{bmatrix}
\begin{bmatrix}
\\
x_{11} & x_{21} &  x_{31}\\\\
\end{bmatrix}
$$

Which becomes:

$$d\textbf{W}^{[1]} = \boldsymbol{\delta}^{[1]}\textbf{x}^{T}$$

Let's now calculate the partial derivative of the loss $J$ with respect to $\textbf{b}^{[1]}$:

$$\dfrac{\partial J}{\partial{\textbf{b}^{[1]}}} = \dfrac{dJ}{d{\textbf{z}^{[1]}}}
\dfrac{\partial{\textbf{z}^{[1]}}}{\partial{\textbf{b}^{[1]}}}$$

We know that:

$$\dfrac{\partial \textbf{z}^{[1]}}{\partial \textbf{b}^{[1]}} = 
\begin{bmatrix}
    \\
    1 & 0 & 0 & 0 \\\\
    0 & 1 & 0 & 0 \\\\
    0 & 0 & 1 & 0 \\\\
    0 & 0 & 0 & 1 \\\\
\end{bmatrix}$$

Which is just the identity matrix. So the partial derivative just simplfies to become:

$$\dfrac{\partial{J}}{\partial{\textbf{b}^{[1]}}} = \boldsymbol{\delta}^{[1]T}$$


And the gradient $d\textbf{b}^{[1]}$ is therefore just the transpose of the Jacobian, or:

$$
d\textbf{b}^{[1]} = \boldsymbol{\delta}^{[1]}
$$

Which is a $(4,1)$ column vector, and matches the dimensions of $\textbf{b}^{[1]}$.

&nbsp;
## In Summary

Our aim was to calculate the gradients for our cost function $J$ with respect to each of our parameters that we wanted to update using stochastic gradient descent.

$$\bigg( d\textbf{W}^{[1]}, d\textbf{b}^{[1]}, d\textbf{W}^{[2]}, d\textbf{b}^{[2]}, 
d\textbf{W}^{[3]}, db^{[3]} \bigg)$$

Below is a summary of the results:

&nbsp;
#### Third Layer
$$d\textbf{W}^{[3]} =  \delta^{[3]}a^{[2]T}$$

$$d b^{[3]} = 
\delta^{[3]}$$
$$\boldsymbol{\delta}^{[3]} = a^{[3]} - y$$

&nbsp;
#### Second Layer

$$d\textbf{W}^{[2]} = \boldsymbol{\delta}^{[2]}\textbf{a}^{[1]T}$$
$$d\textbf{b}^{[2]} = \boldsymbol{\delta}^{[2]}$$

$$
\boldsymbol{\delta}^{[2]}= 
\textbf{W}^{[3]T}
\delta^{[3]}
*
g'(\textbf{z}^{[2]})
$$

&nbsp;
#### First Layer
$$d\textbf{W}^{[1]} = \boldsymbol{\delta}^{[1]}\textbf{x}^{T}$$
$$d\textbf{b}^{[1]} = \boldsymbol{\delta}^{[1]}$$
$$
\boldsymbol{\delta}^{[1]}= 
\textbf{W}^{[2]T}
\delta^{[2]}
*
g'(\textbf{z}^{[1]})
$$

So all that work for a set of simple, predictable equations. Yes, and I think most people will just apply these equations when implementing a neural network in numpy. Or forgo manually architecting backpropagation and gradient descent and using a useful, automatic package like tensorflow. Which is unfortunate, because despite the arduous process of painstakingly producing these equations, I think I've learned a lot by doing it and I hope you, the reader have as well. Thank you!




















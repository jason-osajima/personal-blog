title: word2vec - Negative Sampling
slug: ns
category: 
date: 2019-12-10
modified: 2019-12-10
tags: machine learning, natural language processing, word2vec

*This is part two in a two-part series on the word2vec. Part one is about CBOW and Skip-Gram and can be found [here](/word2vec). Part two is about negative sampling.*

## Introduction

<!-- PELICAN_BEGIN_SUMMARY -->
In the [original word2vec paper](https://papers.nips.cc/ paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), the authors introduced Negative Sampling, which is a technique to overcome the computational limitations of vanilla Skip-Gram. Recall that in the [previous post](/ns), we had a vocabulary of 6 words, so the output of Skip-Gram was a vector of 6 binary elements. However, if we had a vocabulary of, say 170,000 words, we'd find it difficult to compute our loss function for every step of training the model. 

In this post, we will discuss the changes to Skip-Gram using negative sampling and update our Tensorflow word2vec implementation to use it.
<!-- PELICAN_END_SUMMARY -->

## Problem Setup
Let's use the same corpus of documents that we had in the [previous post](/ns): 

    corpus = [document_1, document_2]

The documents are just one sentence long:

    document1 = ["the", "cat", "loves", "fish"]
    document2 = ["the", "person", "hates", "fish"]

And we are still trying to learn the embeddings for the following words in our vocab:
    
    word_to_ix = {'the': 0, 'cat': 1, 'loves': 2, 'fish': 3, 'person': 4, 'hates': 5}

The input to Skip-Gram is a word, and the output is the words that surround that word (called the context). So we would expect to see the following as a training example:

	"cat", ["the", "loves"]

We convert these to indices:

	1, [0, 2]

And convert the output to a binary encoded vector:

	1, [1,0,1,0,0,0]

If we use negative sampling, we will convert this training example into two training examples like so:

	("cat", "the"), 1
	("cat", "loves"), 1

So now the model takes as input a word, context pair $(w, c)$ and attempts to predict whether or not this pair came from the training data (1 if it is, 0 if it is not). If we used this approach, the training data would be quite imbalanced (since it only has positive examples). So how do we get the negative examples? We sample them (hence, negative sampling)!

Which distribution do we sample them from? The [original paper](https://papers.nips.cc/ paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) recommends using the Unigram Model raised to the $3/4$ power. The rationale behind using $3/4$ this can be explained using the example from this [lecture](https://github.com/stanfordnlp/cs224n-winter17-notes/blob/master/notes1.pdf):

> is: $0.9^{3/4}$ = 0.92
> Constitution: $0.09^{3/4}$ = 0.16
> bombastic: $0.01^{3/4}$ = 0.032

So the original probability from the Unigram Model for the word "is" is $0.9$. After taking that probability to the $3/4$ power, its new probability is $0.92$. Not much of a difference. But now look at the word "bombastic". Its probability increases from $0.01$ to $0.032$, a 3x difference. So taking the probabilities of words to the $3/4$ power is a way to normalize probabilities, so that words that show up more infrequently have a higher probability of being sampled.

### What is the Unigram Model?
The Unigram Model is a probability distribution for words that makes the assumption that the words in a sentence are completely independent from one another. So a sentence's probability of occurring is dependent on the probabilities of each of the words in that sentence. Under the unigram model, we'd expect this sentence to have a higher probability:

> "is and and she"

Compared to this sentence:

> "The cantankerous curmudgeon is irascible."

Why does the first sentence have a higher probability of occurring according to the Unigram Model? We want to calculate the probability of the sentence, which is the probability that the sequence of words will occur, $P(is, and, and she)$. Remember that the Unigram Model assumes that words occurences in a sequence are independent of one another, so this probability becomes:

$$P(is, and, and, she) = P(is)P(and)P(and)P(she)$$

Comparing this to the probability of our second sentence:

$$P(the, cantankerous, curmudgeon, is, irascible) = P(the)P(cantankerous)P(curmudgeon)P(is)P(irascible)$$

And it becomes clear that the probability of the first sentence occurring is much higher, because we would expect the probabilities of the rarer words in the second sentence to be much lower than all of the words in the first sentence.

Let's implement the Unigram Model using python. We start by counting the frequency for each word and saving this in a `dict`:

	from collections import defaultdict

	wordFreq = defaultdict(int)

	for document in corpus:
	    for word in document:
	        wordFreq[word] += 1

The result is a frequency dict, which shows the number of times a word showed up in our corpus:

	wordFreq = {'the': 2, 'cat': 1, 'loves': 1, 'fish': 2, 'person': 1, 'hates': 1}

Next, let's convert these frequencies to probabilities. If for a given word $w_i$ the frequency that it shows up in the corpus is $f(w_i)$, then the sample probability for $w_i$ for our distribution will be:

$$P(w_i) = \dfrac{f(w_i)}{\sum^n_{j=0}f(w_j)}$$

Taking the advice from the word2vec authors, we replace $f(w_i)$ with $f(w_i)^{3/4}$:

$$P(w_i) = \dfrac{f(w_i)^{3/4}}{\sum^n_{j=0}f(w_j)^{3/4}}$$

Implementing this in python:

	totalWords = sum([freq**(3/4) for freq in wordFreq.values()])
	wordProb = {word:(freq/totalWords)**(3/4) for word, freq in wordFreq.items()}

Great! Now we can use `np.random.choice` to sample from this probability distribution, and we can use that to generate our negative word, context pairs to use to train our model.

	import numpy as np

	def generate_negative_sample(wordProb):
	    """
	    This function takes as input a dict with keys as the 
	    words in the vocab and values as the probabilities.
	    Probabilities must sum to 1.
	    """    
	    word, context = (np.random.choice(list(wordProb.keys()), 
	                     p=list(wordProb.values())) for _ in range(2))
	    return word, context

	word, context = generate_negative_sample(wordProb)

How many should we generate? Good question. `gensim`, the most popular NLP library in Python, uses a rate of $0.75$ but that might not be the [best rate for all word2vec applications](https://github.com/RaRe-Technologies/gensim/issues/2090). Let's stick with 50% negative samples for this simple example.

	posTrainSet = []

	# add positive examples
	for document in corpus:
	    for i in range(1, len(document)-1):
	        word = word_to_ix[document[i]]
	        context_words = [word_to_ix[document[i-1]], word_to_ix[document[i+1]]]
	        for context in context_words:
	            posTrainSet.append((word, context))

	n_pos_examples = len(posTrainSet)

	# add the same number of negative examples
	n_neg_examples = 0
	negTrainSet = []

	while n_neg_examples < n_pos_examples:
	    (word, context) = generate_negative_sample(wordProb)
	    # convert to indicies
	    word, context = word_to_ix[word], word_to_ix[context]
	    if (word, context) not in posTrainSet:
	        negTrainSet.append((word, context))
	        n_neg_examples += 1

	X = np.concatenate([np.array(posTrainSet), np.array(negTrainSet)], axis=0)
	y = np.concatenate([[1]*n_pos_examples, [0]*n_neg_examples])

Notice that when we generate negative examples, we check if that negative example is the same as a positive example. If it is, we discard it. That makes sense, since we don't want a word, context pair to be both a positive and negative example.

Now, let's initialize the embeddings. We'll change the `input_shape` parameter from `1` to `2`, since we are taking a word and context as inputs into the model instead of just the word.

	N_WORDS = len(word_to_ix.keys())
	embedding_layer = layers.Embedding(N_WORDS, EMBEDDING_DIM, 
	                                   embeddings_initializer="RandomNormal",
	                                   input_shape=(2,))

Next, let's define the model, compile it, and fit it. The only change we make is that we want the output to be a probability that the word, context pair came from the train dataset, so we change the output dimensions to be 1 and the activation to be sigmoid.

	from tensorflow import keras
	from tensorflow.keras import layers
	from keras.models import Model

	model = keras.Sequential([
	  embedding_layer,
	  layers.GlobalAveragePooling1D(),
	  layers.Dense(1, activation='sigmoid'),
	])

	model.compile(optimizer='adam',
	              loss='binary_crossentropy',
	              metrics=['accuracy'])

	history = model.fit(X,y, batch_size=X.shape[0])

And that's a basic implementation of negative sampling. We typically don't want to do negative sampling manually, so luckily gensim and tensorflow do it automatically (however at the time of this post we are [still waiting](https://github.com/tensorflow/tensorflow/issues/34131) for an implementation in the tensorflow keras api).


### Resources
1) [Notes from Stanford NLP course](https://github.com/stanfordnlp/cs224n-winter17-notes/blob/master/notes1.pdf)

2) [word2vec paper](https://arxiv.org/pdf/1301.3781.pdf)

3) [word2vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)









	

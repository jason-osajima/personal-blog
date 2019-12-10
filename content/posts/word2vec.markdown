title: word2vec - CBOW and Skip-Gram
slug: word2vec
category: 
date: 2019-12-09
modified: 2019-12-09
tags: machine learning, natural language processing


## Introduction

<!-- PELICAN_BEGIN_SUMMARY -->
word2vec is an iterative model that can be used to create embeddings of words (or embeddings of [pretty much anything](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)). In this post, we will talk briefly about why you would want to use word2vec, break down the Continuous Bag of Words (CBOW) and skip gram word2vec model, and implement it in tensorflow.
<!-- PELICAN_END_SUMMARY -->

### Why use word2vec?
For a lot of machine learning tasks, we need to figure out the relationship between items. Take for example an NLP task where we use words as items. We somehow need to learn numerical representations of words so that the model can understand the relationship between words. The most straightforward way of representing words numerically would be to represent them as vectors. So, imagine that we trained a model and learned vectors for each word. We'd expect that the word "cat" and "dog" would be close in distance since they are both pets, and that "cat" would be far from "gym", because there isn't much of a relationship between these two words. We sometimes call these vectors embeddings.

### What is word2vec?
word2vec is a model that attempts to learn these embeddings based on a corpus of text (or a group of items). The basic idea is that we initialize a vector for each word in the corpus with random numbers. We then iterate through each word of each document (a document is just a group of words that are related), grab the vectors of the closest n-words on either side of our target word, concatenate these vectors, forward propagate it through a linear layer + softmax function, and attempt to predict what our target word was. We then backpropagate the error between our prediction and the actual target word, and update not only the weights of the linear layer, but also the vectors (or embeddings) of our neighbor words.

Imagine that we have a corpus of two documents:

    corpus = [document_1, document_2]

The documents are just one sentence long:

    document1 = ["the", "cat", "loves", "fish"]
    document2 = ["the", "person", "hates", "fish"]

So the goal of using word2vec is to learn embeddings for all of the words in our corpus. In this case, the words in our corpus are:
    
    {'the': 0, 'cat': 1, 'loves': 2, 'fish': 3, 'person': 4, 'hates': 5}

We can create our vocab `word_to_ix` using the following code:

    def corpus_to_vocab(corpus):
        """
        Takes a corpus (list of documents) and converts
        it to two dictionaries:
          - word_to_ix: key are words in vocab, values
            are the unique indices
          - ix_to_word: key are the unique indices,
            values are the words in vocab
        """
        word_to_ix, ix_to_word = {}, {}
        ix = 0
        for document in corpus:
            for word in document:
                if word not in word_to_ix.keys():
                    word_to_ix[word], ix_to_word[ix] = ix, word
                    ix += 1
        return word_to_ix, ix_to_word
            
    EMBEDDING_DIM = 3

    document1 = ["the", "cat", "loves", "fish"]
    document2 = ["the", "person", "hates", "fish"]
    corpus = [document1, document2]

    # vocab
    word_to_ix, ix_to_word = corpus_to_vocab(corpus)

And we can instantiate embeddings for each of these words in a matrix where the number of rows are equal to the number of words in our vocab and the number of columns is the number of dimensions of our embedding vector. Let's make it simple and work with 3 dimensional embeddings.

$$\textbf{V} = \begin{bmatrix}
  1 & 1 & 4 \\
   5 & 5 & 1 \\
   2 & 1 & 5
\end{bmatrix}$$

If we were interested in looking up the embedding for the word `the`, we would lookup `the` in our `vocab`, get the index $0$, and return the embedding `[1, 1, 4]`.

Now that we've explained the setup of word embeddings, how do we learn them? There are two main ways to learn embeddings using word2vec: Continuous Bag of Words (CBOW), and skipgram. We'll start with explaining CBOW.

## Continuous Bag of Words (CBOW)

The first step for implementing CBOW is to instantiate the embedding matrix described above. Let's create an embedding matrix for the words in our vocab using tensorflow.

    from tensorflow import keras
    from tensorflow.keras import layers

    N_WORDS = len(word_to_ix.keys())
    embedding_layer = layers.Embedding(N_WORDS, EMBEDDING_DIM, 
                                       embeddings_initializer="RandomNormal",
                                       input_shape=(2,))

We define an `input_shape` because the embedding layer is the first layer of our model. The reason why it has an input shape of $(2,)$ is that for each target word, we pass in as input two context words, represented as indices. So if we wanted to pass in "the" and "loves", we would pass in the vector `[0,2]`. 

Great, so now we have an embedding matrix. As an example, we can look up the embedding for `the` by passing a `0`:

    embedding_layer(0)

Next we need to setup our training set for CBOW. Our output is a word in a document, and the input is the context of the word, which is just the n-words to the left and right of the output word. So for example, if we were converting `document1` to our training set, we would end up with:
    
    [(["the", "loves"], "cat"),
     (["cat", "fish"], "loves")]

Notice we don't include `"the"` and `"fish"` in our train set, because we wouldn't have enough context words to construct the training example for them. This isn't a problem when we have a large corpus (but might look like a problem with our small corpus of two document sentences).

We'll convert the words in the train set to indices so it's easy to lookup the word's embeddings:
    
    [([0, 2], 1),
     ([1, 3], 2)]

In code, this looks like:
    
    train_set = []
    for document in corpus:
        for i in range(1, len(document)-1):
            target_word = word_to_ix[document[i]]
            context = [word_to_ix[document[i-1]], word_to_ix[document[i+1]]]
            train_set.append((context, target_word))
    
    X = np.array([example[0] for example in train_set])
    y = np.array([example[1] for example in train_set])
    y = keras.utils.to_categorical(y, num_classes=N_WORDS)

Now that we constructed our train set, let's setup our model.

    model = keras.Sequential([
      embedding_layer,
      layers.GlobalAveragePooling1D(),
      layers.Dense(N_WORDS, activation='softmax'),
    ])

Let's go through the three layers of this model. `embedding_layer` was already described above, and takes as input two indices for the two context words and returns their two embeddings. The output therefore has shape `(None,2,3)`. The first dimension is `None` because it depends on how many words we pass into the model during training. So it could be $1$ if we pass in one training example, and $5$ if we pass in five training examples. `GlobalAveragePooling1d` takes the average of the two embeddings, and its output is  `(None,1,3)`. Finally, `Dense` is a fully-connected layer that multiplies the output of `GlobalAveragePooling1d` by a weight matrix and adds a bias. The resulting vector is then passed through a `softmax` activation, so that we end up with a vector of probabilities for the index of the target output word.

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X,y, batch_size=X.shape[0])

Next, we compile the model, using `categorical_crossentropy` as our loss function since this is a multi-category problem, and then we are done! Obviously, our word embeddings won't be very "good" because we only trained a model on  two sentences. To check how the embeddings for the word `the` changed, we just pass:
    
    embedding_layer(word_to_ix[`the`])

So what changes when we use skipgram?

## Skip-Gram
The main change is that we switch the output and input for the model. So the input into the model is now the target word, and the output are the context words. For example, the input could now be the word `cat`, and the output could be the words `love` and `the`. We represent the input word as its index, so we'd feed in $1$ for the word `cat`. What would the output be? If the context is `love` and `the`, the output would be `[1,1,0,0,0]`. I like to call this multi-hot encoding, but I'm not sure if that's the best term for it. We also change the loss to `binary_crossentropy`, since we now have [converted our problem to multi-label](https://github.com/keras-team/keras/issues/2166).

    train_set = []

    for document in corpus:
        for i in range(1, len(document)-1):
            target_word = word_to_ix[document[i]]
            context = [word_to_ix[document[i-1]], word_to_ix[document[i+1]]]
            train_set.append((target_word, context))

    N_WORDS = len(word_to_ix.keys())
    embedding_layer = layers.Embedding(N_WORDS, EMBEDDING_DIM, 
                                       embeddings_initializer="RandomNormal",
                                       input_shape=(1,))

    X = np.array([example[0] for example in train_set])
    y = np.array([example[1] for example in train_set])
    y = keras.utils.to_categorical(y, num_classes=N_WORDS)
    y = np.sum(y, axis=1).astype('int')

    model = keras.Sequential([
      embedding_layer,
      layers.GlobalAveragePooling1D(),
      layers.Dense(N_WORDS, activation='softmax'),
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X,y, batch_size=X.shape[0])

### Conclusion
So we now know how to implement CBOW and SkipGram word2vec in tensorflow. Hooray! Don't get too excited though: these implementations are not very practical. The reason is because the number of words will not typically be 6, like in our example. Let's imagine that we implemented word2vec using all of wikipedia. Wikipedia has 2.9 billion words. We can't have a 2.9 billion one-hot or multi-hot encoded output. Luckily, we have a way to get around this by using negative sampling, which was introduced in the [original word2vec paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).


### Resources
1) [Notes from Stanford NLP course](https://github.com/stanfordnlp/cs224n-winter17-notes/blob/master/notes1.pdf)

2) [word2vec paper](https://arxiv.org/pdf/1301.3781.pdf)

3) [difference between skipgram and cbow](https://stackoverflow.com/questions/38287772/cbow-v-s-skip-gram-why-invert-context-and-target-words)
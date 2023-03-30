# Text-Emotion-Recognition-Using-TensorFlow-RNN
# 1. Introduction :

Emotions are highly valued because they play an important role in human
interaction. A model is created and trained to recognize emotions in text using
Recurrent Neural Networks. The dataset is divided into six categories: love, fear,
joy, sadness, surprise, and anger.

# 2. Tools used:

## TensorFlow

TensorFlow is a free and open-source software library for dataflow and
differentiable programming across a range of tasks. It is a symbolic math library,
and is also used for machine learning applications such as neural networks. It is used
for both research and production at Google.

## Keras

Keras is an open-source neural-network library written in Python. It is capable of
running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML.
Designed to enable fast experimentation with deep neural networks, it focuses on
being user-friendly, modular, and extensible. It was developed as part of the
research effort of project ONEIROS (Open-ended Neuro-Electronic Intelligent Robot
Operating System), and its primary author and maintainer is François Chollet, a
Google engineer. Chollet also is the author of the XCeption deep neural network
model.

Keras is based on minimal structure that provides a clean and easy way to create
deep learning models based on TensorFlow or Theano. Keras is designed to quickly
define deep learning models. Well, Keras is an optimal choice for deep learning
applications.

## NumPy

NumPy is a library for the Python programming language, adding support for
large, multi-dimensional arrays and matrices, along with a large collection of high-


level mathematical functions to operate on these arrays. The ancestor of NumPy,
Numeric, was originally created by Jim Hugunin with contributions from several
other developers. In 2005, Travis Oliphant created NumPy by incorporating
features of the competing Numarray into Numeric, with extensive modifications.
NumPy is open-source software and has many contributors.

NumPy targets the CPython reference implementation of Python, which is a non-
optimizing bytecode interpreter. Mathematical algorithms written for this version of
Python often run much slower than compiled equivalents. NumPy addresses the
slowness problem partly by providing multidimensional arrays and functions and
operators that operate efficiently on arrays, requiring rewriting some code, mostly
inner loops, using NumPy.

## Pandas

Pandas is a Python library for data analysis. Started by Wes McKinney in 2008
out of a need for a powerful and flexible quantitative analysis tool, pandas has
grown into one of the most popular Python libraries. It has an extremely active
community of contributors.

Pandas is built on top of two core Python libraries—matplotlib for data
visualization and NumPy for mathematical operations. Pandas acts as a wrapper
over these libraries, allowing you to access many of matplotlib's and NumPy's
methods with less code. For instance, pandas' .plot() combines multiple matplotlib
methods into a single method, enabling you to plot a chart in a few lines.

## Seaborn

Seaborn is an open-source Python library built on top of matplotlib. It is used for
data visualization and exploratory data analysis. Seaborn works easily with
dataframes and the Pandas library. The graphs created can also be customized
easily. Below are a few benefits of Data Visualization.

## Matplotlib

Matplotlib is a plotting library for the Python programming language and its
numerical mathematics extension NumPy. It provides an object-oriented API for
embedding plots into applications using general-purpose GUI toolkits like Tkinter,
wxPython, Qt, or GTK+. There is also a procedural "pylab" interface based on a
state machine (like OpenGL), designed to closely resemble that of MATLAB, though


its use is discouraged. SciPy makes use of Matplotlib. Several toolkits are available
which extend Matplotlib functionality. Some are separate downloads, others ship
with the Matplotlib source code but have external dependencies.

Pyplot is a Matplotlib module which provides a MATLAB-like interface. Matplotlib is
designed to be as usable as MATLAB, with the ability to use Python and the
advantage of being free and open-source. Each pyplot function makes some
change to a figure: e.g., creates a figure, creates a plotting area in a figure, plots
some lines in a plotting area, decorates the plot with labels, etc. The various plots
we can utilize using Pyplot are Line Plot, Histogram, Scatter, 3D Plot, Image,
Contour, and Polar.

## Pickle

Python pickle module is used for serializing and de-serializing a Python object
structure. Any object in Python can be pickled so that it can be saved on disk. What
pickle does is that it “serializes” the object first before writing it to file. Pickling is a
way to convert a python object (list, dict, etc.) into a character stream. The idea is
that this character stream contains all the information necessary to reconstruct the
object in another python script.

## Sklearn

Scikit-learn (Sklearn) is the most useful and robust library for machine learning in
Python. It provides a selection of efficient tools for machine learning and statistical
modeling including classification, regression, clustering and dimensionality reduction
via a consistence interface in Python. This library, which is largely written in Python,
is built upon NumPy, SciPy and Matplotlib.

# 3. Dataset:

The dataset consists of (20,000 rows) Collection of documents and its emotions, it
helps greatly in NLP Classification tasks. The dataset is already pre-processed and
divided into the training, test and validation set. The training set consists of 16,
rows, the test set consists of 2,000 rows and the validation set also consists of 2,
rows.


The dataset used can be downloaded from
https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp

# 4. Methodology:

The RNN model was created using the Google Collab environment, which hosts the
Jupyter notebook in the cloud. TensorFlow is used as the machine learning
framework. First, all of the required libraries are imported. The dataset is then
imported and assigned to the appropriate data object. TensorFlow's built-in
tokenizer is used for text pre-processing, and each word in the dataset is assigned
to a unique token. Following that, the tokens are padded and truncated so that the
model receives input with a fixed shape. Then we create a dictionary to convert the
class names to their corresponding indexes. The text labels for the various classes
are passed in order to obtain numeric representations of them. Five different layers
are used to create the sequential model. After that, the model is trained and
evaluated.

## a. Installing text_hammer and genism:

- The package text_hammer is used to facilitate text preprocessing. It installed
    using the following command:
    !pip install text_hammer
- Gensim is **a free open-source Python library for representing documents as**
    **semantic vectors** , as efficiently (computer-wise) and painlessly (human-wise)
    as possible. Gensim is designed to process raw, unstructured digital texts
    (“plain text”) using unsupervised machine learning algorithms. It is installed
    using the following command:
    !pip install gensim (or, alternatively for conda environments:
    conda install -c conda-forge gensim)


## b. Importing the libraries:

**import tensorflow as tf**

**import numpy as np**

**import pandas as pd**

**import matplotlib.pyplot as plt**

**import seaborn as sns**

**import pickle**

**import sklearn**

**import warnings** (To hide some unnecessary warnings shown by Jupyter)

Further some modules from the Tensorflow library are also imported.

## c. Importing the Dataset

The Emotion Dataset is imported from drive. The dataset as said before is already
divided into test, training and validation sets. Each set has text and label features.

## d. Text preprocessing:

In order to make our text data cleaner we need to perform some text
preprocessing.


- Removing punctuations (it doesn’t contribute to emotion detection).
- Removing stopwords ( i.e. words like the, are, etc. which also does not
    contribute to the task).
- Removing emails, HTML tags, website, and unnecessary links.
- Removing contraction of words ( I’m -> I am ).
- Normalisation of words ( eating -> eat, playing -> play).

## e. Label Encoding:

The sentiment category in our data frame needs to be converted into some numbers
in order to pass into the model.

Using a dictionary we are encoding our sentiment
categories **{‘joy’:0,’anger’:1,’love’:2,’sadness’:3,’fear’:4,’surprise’:5}.**

encoded_dict = {'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}

## f. Tokenization:

TensorFlow includes a Tokenizer library that is imported from its text pre-processing
module. Tokenization generates a token value at random for plain text and saves
the mapping in a database. Document words must be tokenized so that each word
can be represented as a number and fed into the model, allowing the model to
train on the data. The tokenizer basically creates a corpus (collection) of all the
words in the dataset and assigns a unique token to each unique word.

A limit is also set on how many of the most frequently used words are to be
organized, and the remaining less frequently used words are given a common token
called out of vocabulary, which is essentially an unknown word token.

An object tokenizer is created that tokenizes the top 10,000 most frequently used
words in the text corpus and assigns an unknown token () to the remaining words.
The words from the training set are then mapped to numeric tokens using the fit on
texts function.


## g. Word2Ve:

Before proceeding to the next step, you need to look back to the last step there is
one problem in our approach.

Let’s say we have words (‘love’, ‘affection’,’ like’) these words have the same
meaning but according to our tokenizer these words are treated differently. we
need to create a relationship between all those words which are interrelated.

## h. Creating the Model:

Keras is used to create a sequential model. Recurrent Neural Network (RNN) is a
deep learning algorithm designed for sequential data. In an RNN, the neural
network learns from the previous step in a loop. The output of one unit is fed into
the next, and information is passed along.

However, RNNs are not suitable for training large datasets. During RNN training,
the information is looped repeatedly, resulting in very large updates to neural
network model weights, which leads to the accumulation of error gradients during
the update and the network becoming unstable. At the extreme, weight values can
become so large that they overflow and result in NaN values. The explosion occurs
through exponential growth, which is achieved by repeatedly multiplying gradients
through the network layers with values greater than one, or vanishing if the values
are less than one.

To solve this issue, Long Short-Term Memory (LSTM) is employed. Long-term
dependencies can be captured by LSTM. It can remember previous inputs for
extended periods of time. An LSTM cell has three gates: forget, input, and output.

- Forget Gate: The forget gate removes information from the cell that is no
    longer useful.
- Input Gate: The input gate adds additional useful information to the cell
    state.
- Output Gate: The output gate adds additional useful information to the cell
    state.


These gates are used in LSTM memory manipulation. Long short-
term memory (LSTM) employs gates to control gradient
propagation in the memory of a recurrent network. This LSTM
gating mechanism has allowed the network to learn when to
forget, ignore, or keep information in the memory cell.

The model's first layer is the Embedding layer. It has an input
dimension of 10,000 (the most frequently used words in the
dataset) and an output dimension of 100 (the size of the output
vectors from this layer for each word). The sequence's input length
will be the maximum length of 300.

LSTM uses the hidden state to save information from previously processed inputs.
Because the only inputs it has seen are from the past, unidirectional LSTM only
preserves information from the past. Bidirectional LSTM will run the inputs in two
directions, one from past to future and one from future to past. What distinguishes
this approach from unidirectional is that in the LSTM that runs backwards,
information from the future is preserved, and by combining the two hidden states, it
is possible to preserve information from both past and future at any point in time.

The second layer is a bidirectional LSTM. This means that the contents of the LSTM
layer can be read from left to right as well as right to left. Its 100 cells (each with
its own inputs, outputs, and memory) are used, and the return sequence is set to true,
which means that whenever an output is fed into another bidirectional LSTM layer, it
is sent as a sequence rather than a single value of each input, so that the
subsequent LSTM layer can have the required input.

The final layer will be a Dense layer with six units for each of the six classes
present, and the activation will be set to softmax, which will return a probability
distribution over the target classes.

The Dropout layer is added to a model between existing layers and applies to
outputs of the prior layer that are fed to the subsequent layer. Dropout is
a technique used to prevent a model from overfitting.

Because the classes are not one-hot encoded, the model is compiled with the loss set
‘categorical_crossentropy’ (for binary classes). The optimizer used is ‘adam’, which
is extremely efficient when dealing with large datasets. The ‘accuracy’ metric is
used for training, and it calculates how often the predictions match the actual labels.


## i. Training the Model:

The validation set has been prepared, and the sequences for it have been
generated. Its labels are also converted to their numerical representation.

After that, the model is trained for 25 epochs. The number of epochs is a gradient
descent hyperparameter that controls the number of complete passes through the
training dataset. An early stopping callback is also set, which stops the training if
the model does not improve its validation accuracy after 5 epochs. The model
training took us about 6 hours to complete because the Jupyter notebook service is
hosted on Google Colab, which uses a GPU for accelerated computation, as
opposed to the Jupyter in conda environment, which took 1 hour to finish just one
epoch.

## j. Evaluating the Model:

Using training history analyzing the model performance.

The model achieves 92 % accuracy on the test set which is very similar to the
accuracy achieved on the validation dataset.


## k. Outcome:

We tested some texts to predict the emotion of each and comparing it with its
actual emotion.

**Text:** i feel crappy i eat crappy

**Actual Emotion:** sadness

**Predicted Emotion:** sadness

**Text:** i feel really irritated when i talk about my problems and people start talking
about theirs

**Actual Emotion:** anger

**Predicted Emotion:** anger

**Text:** i woke up and felt sad all over again but that was quickly replaced with a
feeling that reassured me things will work themselves out on their own time

**Actual Emotion:** joy

**Predicted Emotion:** sadness

**Text:** i am feeling overwhelmed by trying to do it all that i think on the women
before me

**Actual Emotion:** surprise

**Predicted Emotion:** fear

**Text:** i finally arrived home a couple of hours later feeling somewhat exhausted
dehydrated and even sun burnt

**Actual Emotion:** sadness

**Predicted Emotion:** sadness


**Text:** i feel useful and valued and that is fundamental for me

**Actual Emotion:** joy

**Predicted Emotion:** joy

**Text:** i am feeling outraged it shows everywhere

**Actual Emotion:** anger

**Predicted Emotion:** anger

**Text:** i still sit back and feel amazed by the whole thing

**Actual Emotion:** surprise

**Predicted Emotion:** surprise

## l. Conclusion:

In this project, a RNN model is constructed to recognize the emotions in tweets. The
Model produces an accuracy rate of about 92.75%.

```
Training Validation
Accuracy 94. 46 % 92. 75 %
Loss 12. 61 % 19. 29 %
```
All the predictions are also evaluated against all the ground truths using the test set.
A confusion matrix is generated for the test labels in the against the actual classes.


The model produces mostly accurate results, but based on the confusion matrix, the
most common misclassification appears to be joy and love classes, as well as fear
and surprise classes. This can be corrected by balancing the number of data across
all emotions.

In the future, a much larger dataset with more epochs can be used to improve the
accuracy.



			Question Type Classification


List of contents:


* Introduction
* Softwares
* Procedure
* Libraries
* Feature extraction techniques
* Implementation and results
* Credits/Contributors


Introduction:
        
Question type classification is a process of classifying the labels to the respective questions. 
According to the given corpus, it contains eight different labels such as Person, Money, Organisation, Location, Number, Date, Time, Percentage. 
And the dataset contains 1340 queries which are labelled according to respective categories. Labelled dataset containing text documents and their labels is used to train a classifier.


        This classification type can be done by using various approaches such as statistical and neural network based. 


Softwares used:


Users can implement the code in Jupyter notebook or in google colaboratory(colab).


Procedure for classification:


There are mainly four steps in performing the classification:


1. Dataset Preparation: The first step is the Dataset Preparation step which includes the process of loading a dataset and performing basic pre-processing. 
The dataset is then splitted into train and validation sets.
2. Feature Engineering: The next step is Feature Engineering in which the raw dataset is transformed into flat features which can be used in a machine learning model. 
This step also includes the process of creating new features from the existing data.
3. Model Training: The final step is the Model Building step in which a machine learning model is trained on a labelled dataset.
4. Improve Performance of Classifier: we can improve the performance by using various models.


Libraries:


We need to import some libraries for implementing the classification. They are:
* Pandas
* Scikit-learn
* Keras
* Numpy
* Transformers


Feature Extraction Techniques:
In order to give the data as input to the statistical models,Deep learning neural networks, data has to be transformed into flattened features so that we can give that flattened features as input to the models.
We transformed the data into TFIDF(Term frequency-inverse document frequency) vectors using 3 different types(word level, char level, n-gram level) which can be used as input features for statistical models. 
We transformed the data into an embedding matrix using pretrained FastText and BytePair embeddings which can be used as input features for Deep learning neural networks.




Statistical approach: 


Statistical modeling is the process of applying statistical analysis to a dataset.
 A statistical model is a mathematical representation (or mathematical model) of observed data. 
In this approach we have different models such as support vector machine, Logistic regression, Multilayer perceptron(mlp), Naive bayes, Random Forest, Decision tree.


Support Vector Machine:


support-vector machines are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. 


   WordLevel tfidf      -  80
   N-gram level tfidf   -  72
   Char-level tfidf     -  83
   Count level vectors  -  74




Logistic regression:


Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic/sigmoid function. 


   WordLevel tfidf      -  81
   N-gram level tfidf   -  71
   Char-level tfidf     -  82
   Count level vectors  -  78




Multilayer Perceptron:


A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). ... An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.


   WordLevel tfidf      -  80
   N-gram level tfidf   -  71
   Char-level tfidf     -  86
   Count level vectors  -  74




Naive Bayes:


Naive Bayes is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors.
 A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature here.


   WordLevel tfidf      -  73
   N-gram level tfidf   -  73
   Char-level tfidf     -  77
   Count level vectors  -  76




Random Forest:


Random Forest models are a type of ensemble models, particularly bagging models. They are part of the tree based model family.


   WordLevel tfidf      -  80
   N-gram level tfidf   -  76
   Char-level tfidf     -  84
   Count level vectors  -  77










Decision Tree:


Decision Trees are a non-parametric supervised learning method used for classification and regression. 
The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.


   WordLevel tfidf      -  71
   N-gram level tfidf   -  71
   Char-level tfidf     -  75
   Count level vectors  -  71






Neural Network Approach:


Deep Neural Networks are more complex neural networks in which the hidden layers perform much more complex operations than simple sigmoid or relu activations. 
Different types of deep learning models can be applied in text classification problems.


We have used two types of embeddings in this classification. They are:


* FastText
* Byte-Pair Encoding (BPEmb) 


One can find the pretrained telugu FastText and Byte Pair Word embeddings in the following websites:
  
   https://fasttext.cc/docs/en/crawl-vectors.html  - FastText Word embeddings
   https://bpemb.h-its.org/te/                     - Byte Pair Word embeddings




Here, we have implemented the using various neural networks such as


* Convolutional Neural Network (CNN)
* Long Short Term Modelr (LSTM)
* Gated Recurrent Unit (GRU)
* Bidirectional RNN
Convolutional Neural Network
In Convolutional neural networks, convolutions over the input layer are used to compute the output. 
This results in local connections, where each region of the input is connected to a neuron in the output. Each layer applies different filters and combines their results.
   FastText word embeddings      -  84
   Byte pair word embeddings     -  82






Recurrent Neural Network – LSTM
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. ... 
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.
   FastText word embeddings      -  83
   Byte pair word embeddings     -  83




Recurrent Neural Network – GRU
Gated recurrent units (GRUs) are a gating mechanism in recurrent neural networks, introduced in 2014 by Kyunghyun Cho et al. 
The GRU is like a long short-term memory (LSTM) with a forget gate, but has fewer parameters than LSTM, as it lacks an output gate.


   FastText word embeddings      -  84
   Byte pair word embeddings     -  84




Bidirectional RNN
Bidirectional recurrent neural networks connect two hidden layers of opposite directions to the same output. 
With this form of generative deep learning, the output layer can get information from past and future states simultaneously.


   FastText word embeddings      -  84
   Byte pair word embeddings     -  84




Bidirectional Encoder Representations(BERT)


Bidirectional Encoder Representations from Transformers is a transformer-based machine learning technique for natural language processing pre-training developed by Google.


Accuracy obtained with BERT  - 85


Credits/Contributors:


Under the Mentorship of :
URLANA ASHOK (LRTC IIIT Hyderabad)


Implemented by:
* Jishnu Sai Sukesh
* Siva Prasad Reddy
* Dharani Priya
* Priyanka
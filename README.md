# Report, questions and documentation
## Exploratory analysis findings
_/Notebook/exploratory_analysis.ipynb_

During a preliminary exploration of the dataset some info emerged:
- 1.248.417 number of entry point
- about 1.5% of the product id (asin) is duplicate 
- half of the columns have a high number of missing values: also_buy (72%), also_view (65%), image (48%), price (47%), feature (29%) 

_price column_ 
- studying the price distribution grouped by category doesn't show any particular information, only the Automotive 
category reveal an average price higher than the others
- the max price in many categories is 999, suggesting the presence of an upper bound in the value

_description column_
- the average number of words in the description field is 94, with the 25% and the 75% percentiles respectively set to 27 and 
121 words
- the min and max values suggest the presence of outliers since there are descriptions with only one word and other with 
10.750

_main category column_
- it doesn't seem to be a significant class imbalance. The class with less entry points is 'Health & Personal Care' category
with about 10k values
- the study of the missing values within the different category doesn't show any particular imbalance, except for the 
columns brand (2 category with around 30% of missing data) and feature (6 category with more than 80% of missing values)

_brand column_
- the number of brands that operate in more than one category are 35.073 (13% of the total)
- having a look at the brand names it is possible to see some values like ‘Unknown’, ‘None’, ‘Other’, etc; which might 
indicate the presence of missing values not considered in the initial case (only "" and “[]”) but since I cannot 
contact any expert of domain it is not possible to exclude them a priori (a brand could be called ‘Unknown’).

### Some reasoning after the first analysis of the dataset
Without the initial investigation it would have been possible to approach the problem in several ways:
- classification of the category using a CNN with the images of the product
- GNN to classify the node of the network from the products seen and bought 
- retrieve some useful information about the category from the price amount

But with the problems shown in the previous paragraphs those approach seems unrealisable.  
Given this, and excluding any possibility of scraping information from the web (out of the problem scope), the remaining
possibility is using NLP techniques to infer the category of the product from the title and the description

## Approach followed

I decided to approach the problem in 3 different ways, and then evaluate the best solution based on trade-offs:
- BagOfWord with TF-IDF: first use the TF-IDF algorithm to weight all the corpus and then train a classifier to predict the category
- Embedding FastText: it is possible to use the pretrained embedding space of FastText as first layer of a NN to classify the category
- Transfer learning with BERT: using BERT with the precalculated weights, and adapting the last layers for the problem, to classify the category

All those approach and their performances can be found inside the folder /Notebooks 
- _tf-idf_model_
- _embedding_model_
- _bert_transfer_learning_

## Models comparison
Without being aware of the business scenario in which this problem fits, it is difficult to extract metrics to optimize. 
For this reason, I have tried to imagine a practical use case in which to use the trained predictive model. 

My idea is to support sellers of ecommerce websites in the automatic assignation of the category of their products (a task often complicated even for humans) from the description of them.
In this scenario the parameter to optimize is the score in which the model assigns the correct category among all the products under consideration.  
This translates to the total accuracy of the model as a metric to optimize. In addition to that, in case of models with equal accuracy, it is preferred the model with average f1 score, among classes, higher

__Models performance__
- TF-IDF: 73% accuracy 
- Embedding method: 75% accuracy
- Transfer learning with BERT: 60% accuracy, the model has been trained only for 4 epochs cause of the time limitation (took hrs even using powerful GPUs)

Given the accuracy and f1-scores, and considering the weight of the final model, it has been chosen to use as model for the inference the second one.  
Having a look at the confusion matrix from the prediction over the test set, it is possible to see how the category mostly misclassified was the 'All Electronics'.  
Specifically, from the 9000 'All electronics' products in the test set, more than 2000 have been classified as 'Computers' and almost 2500 as 'Home Audio'. It is easily understandable how difficult could be to assign those products to the correct class, even for a human.   

## Further implementations
- Analysis of other approaches:  
  - Classification based on the field also_buy and also_view, basically counting the most frequent linked class or modelling a GNN for node classification
  - CNN for classification of the product images 
- Model stacking: after checking for the non correlation of the models it will be possible to train a meta-learner that takes the output of all the trained model and return the final (and presumably more accurate) prediction
- Use the brand to help the neural network in restricting the possible categories: if a brand have products in only a single category enforce the prediction in that direction
- Better training of the transfer learning model with BERT: due to a time and resources limitation the transfer learning approach forced me 
to limit the number of training epochs to 4, but potentially the model has still room to improve 
- More in depth data exploration: an assumption that I made was that all the text was written in english, but it would be a good practice
to confirm that. If that is the case it would be possible to train a model for every accepted language and execute the only the proper one (after the identification of the language with a model on top)
- Summarize the long descriptions of the products, so it is no longer necessary to trim part of the text and lose information
- Maintain a modular approach (using ML pipelines) with the models, to easily exchange them in the deployment phase


# Questions

__What would you change in your solution if you needed to predict all the categories?__ 

With only the information available at the moment I see two possible ways to approach this change:
- The definition of the field category says: _"list of categories the product belongs to, usually in hierarchical order"_. 
So I would check what does it mean 'usually' in this context. Because if the assumption of hierarchical order holds, I would first build the
"category tree" and then train the model to predict only the leafs, and then return the whole branch of categories. That would lead only to an increase of the total possible classes
- Standard approach in case of multi-label problems: create a model where the output result is given by a label encoding in which if a product correspond to a category it will get 1 in the vector, 0 otherwise (e.g. [0,1,0,0,1,1])

__If this model was deployed to categorize products without any supervision which metrics would you check to detect data drifting? When would you need to retrain?__    

If it is not possible to have supervised categorization to track the standard metrics over time (like accuracy, f1-score and so on), those 
are some possible triggers of a new training for the model:
- average confidence score of the predicted class: if the model is no longer as confident as before in the assignation of the correct class
- change in the distribution of the predicted categories: it might be due to a natural change in the true value of the product categories
but since the supervision is not possible this could signal the need of a new training 
- new categories of products: if a new category is inserted into the system the model will not be able to predict it 
- change in characteristic of the description feature, like: average length of the text increase, products in different languages, etc.

# Documentation
For a proper setup it is required to place the dataset amz_products_small.json.gz in the main directory with the .py scripts.

The script _split_training_test.py_ will populate the folder _/Data_ with the dataset for training (cleaned and preprocessed) and test.

The file _utility.py_ contains a series of general functions used inside the notebooks.

Inside the _/Notebooks_ folder are located all the notebooks described in the previous paragraphs.
Running the notebooks will generate models as outputs, which are saved inside the _/Models_ folder

To run the docker container execute `docker build` and `docker run` commands inside the folder _/Docker_.
That will load the image to perform the inference using the model from the notebook _embedding_model.ipybn_, and for that
it is necessary to put inside the _/model_ folder the outputs generated from the notebook (_embedding_fasttext.joblib and _tokenizer_embedding_fasttext.joblib)

Note. Since it has been used the FastAPI library to deploy the model, it is possible, after having lunched the container, to access the openApi specs via `0.0.0.0:<port>/docs`




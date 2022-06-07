# document-classification
This project uses the probability background to make predictions in uncertain situations, applying concepts like conditional probability, priors, and conditional independence. 

## Summary
We'll be reading in a corpus (a collection of documents) with two possible labels and training a classifier to determine which label a query document is more likely to have.

Here's the twist: the corpus is created from CS 540 essays about AI from 2016 and 2020 on the same topic. Based on training data from each, you'll be predicting whether an essay was written in 2020 or 2016. (Your classifier will probably be bad at this! It's okay, we're looking for a very subtle difference here.)

## Steps
1. Loading the data into a convenient representation.
2. Computing probabilities from the constructed representation.
3. Use the probabilities computed to create a prediction model and use this model to classify test data. 

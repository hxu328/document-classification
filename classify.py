import os
import math

# These first two functions require os operations and so are completed for you
# Completed for you


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset

# Completed for you


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """

    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f, 'r', encoding='UTF-8') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])

# The rest of the functions need modifications ------------------------------
# Needs modifications


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # TODO: add your code here
    with open(filepath, 'r', encoding='UTF-8') as file:
        for word in file:
            word = word.strip()
            if word in vocab and word not in bow and len(word) > 0:
                bow[word] = 1
            elif word in vocab and word in bow and len(word) > 0:
                bow[word] += 1
            elif word not in vocab and None not in bow.keys() and len(word) > 0:
                bow[None] = 1
            elif word not in vocab and None in bow.keys() and len(word) > 0:
                bow[None] += 1
    return bow

# Needs modifications


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1 # smoothing factor
    logprob = {}
    # TODO: add your code here
    total_num_of_files = len(training_data)
    for label in label_list:
        num_of_file_with_label = 0
        for file in training_data:
            if file['label'] == label:
                num_of_file_with_label += 1
        logprob[label] = math.log((num_of_file_with_label + smooth)/(total_num_of_files + 2))

    return logprob

# Needs modifications


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1 # smoothing factor
    word_prob = {}
    # TODO: add your code here

    size_of_vocab = len(vocab)
    total_word_count_given_label = 0
    for file in training_data:
        if file['label'] == label:
            temp_dict = file['bow']
            for single_word in temp_dict:
                total_word_count_given_label += temp_dict[single_word]

    total_none_count_given_label = 0
    for file in training_data:
        if file['label'] == label:
            tep = file['bow']
            if None in tep:
                total_none_count_given_label += tep[None]

    for word in vocab:
        temp_count = 0
        for doc in training_data:
            if doc['label'] == label:
                temp = doc['bow']
                if word in temp:
                    temp_count += temp[word]
        word_prob[word] = math.log((temp_count + smooth * 1)/(total_word_count_given_label + smooth *(size_of_vocab + 1)))

    word_prob[None] = math.log((total_none_count_given_label + smooth * 1)/(total_word_count_given_label + smooth *(size_of_vocab + 1)))


    return word_prob


##################################################################################
# Needs modifications
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    # TODO: add your code here
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    log_prior = prior(training_data, ['2020', '2016'])

    retval['vocabulary'] = vocab
    retval['log prior'] = log_prior
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')

    return retval

# Needs modifications


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    # TODO: add your code here
    file = open(filepath, 'r', encoding='UTF-8')
    prior_2016 = model['log prior']['2016']
    prior_2020 = model['log prior']['2020']
    trained_2016 = model['log p(w|y=2016)']
    trained_2020 = model['log p(w|y=2020)']
    p_2016 = prior_2016
    p_2020 = prior_2020

    with open(filepath, 'r', encoding='UTF-8') as file:
        for word in file:
            word = word.strip()
            if len(word) > 0 and word in trained_2016:
                p_2016 += trained_2016[word]
            elif len(word) > 0 and word not in trained_2016:
                p_2016 += trained_2016[None]

            if len(word) > 0 and word in trained_2020:
                p_2020 += trained_2020[word]
            elif len(word) > 0 and word not in trained_2020:
                p_2020 += trained_2020[None]

    if p_2016 > p_2020:
        retval['predicted y'] = '2016'
    elif p_2020 > p_2016:
        retval['predicted y'] = '2020'
    retval['log p(y=2016|x)'] = p_2016
    retval['log p(y=2020|x)'] = p_2020

    return retval




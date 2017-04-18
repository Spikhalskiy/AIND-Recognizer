import warnings
from operator import itemgetter

from asl_data import SinglesData


def estimate(model, word, x, length, verbose):
    try:
        score = model.score(x, length)
        if verbose:
            print('Recognize success for word "{}", score: {}'.format(word, score))
        return score
    except Exception as er:
        if verbose:
            print('Recognize error for word "{}", error: {}'.format(word, er))
        return None


def recognize(models: dict, test_set: SinglesData, verbose = False):
    """ Recognize test word sequences from word models set

    :param verbose: Print log with debug information
    :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
            ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for sentence_index in range(test_set.num_items):
        x, length = test_set.get_item_Xlengths(sentence_index)

        words_scores = {word_score[0]: word_score[1] for word_score in
                        [(word, estimate(models[word], word, x, length, verbose)) for word in models.keys()] if word_score[1]}
        best_word_score = sorted(words_scores.items(), key=itemgetter(1), reverse=True)[0]
        best_word = best_word_score[0]

        if verbose and not best_word:
            print('We didn''t find best word for sentence {}'.format(sentence_index))

        probabilities.append(words_scores)
        guesses.append(best_word)

    return probabilities, guesses

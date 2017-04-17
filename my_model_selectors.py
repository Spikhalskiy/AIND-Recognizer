import warnings

import math
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

    def create_and_fit(self, num_components, sequences_train, sequences_lengths_train):
        model = self.base_model(num_components)
        if not model:
            if self.verbose:
                print("Model is empty")
            return None
        assert isinstance(model, GaussianHMM)
        model = model.fit(sequences_train, sequences_lengths_train)
        if not model:
            if self.verbose:
                print("Model is empty")
            return None
        assert isinstance(model, GaussianHMM)
        return model

    def fit_and_score(self, num_components, sequences_train, sequences_lengths_train,
                      sequences_test, sequences_lengths_test):
        """
        :return: model score for passed components number, using provided collections for fit and score.
        """

        try:
            model = self.create_and_fit(num_components, sequences_train, sequences_lengths_train)
            if not model:
                return None
            assert isinstance(model, GaussianHMM)
            logL = model.score(sequences_test, sequences_lengths_test)
            return logL
        except Exception as e:
            if self.verbose:
                print("Exception during training model {}".format(e))
            return None

    def select_info_and_produce_model(self, models_info):
        best_model_info = sorted(models_info,
                                 key=lambda info: info.score
                                 if info and info.score else float("-inf"), reverse=True)[0]

        if not best_model_info:
            if self.verbose:
                print("No models left after sorting")
            return None
        assert isinstance(best_model_info, ModelInfo)
        if self.verbose:
            print("Final num of components {}".format(best_model_info.num_components))
        return self.base_model(best_model_info.num_components)


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models_info = [ModelInfo(num_components)
                       for num_components in range(self.min_n_components, self.max_n_components + 1)]
        for model_info in models_info:
            try:
                model_logL = self.fit_and_score(model_info.num_components, self.X, self.lengths, self.X, self.lengths)
                if not model_logL:
                    if self.verbose:
                        print("No score for a model with num_components {}".format(model_info.num_components))
                        continue
                N, features = self.X.shape

                """
                Dana Sheahen April 6th at 10:31 AM  
                There is one thing a little different for our project though...
                in the paper, the initial distribution is estimated and therefore
                those parameters are not "free parameters". 
                However, hmmlearn will "learn" these for us if not provided. 
                Therefore they are also free parameters:
                => p = n*(n-1) + (n-1) + 2*d*n
                       = n^2 + 2*d*n - 1
                """

                p = model_info.num_components ** 2 + 2 * features * model_info.num_components - 1
                bic_score = -2 * model_logL + p * math.log(N)
                model_info.score = bic_score
            except Exception as e:
                if self.verbose:
                    print("Exception during calculation BIC score for n_components: {}, {}"
                          .format(model_info.num_components, e))

        return self.select_info_and_produce_model(models_info)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        models_info = [ModelInfo(num_components)
                       for num_components in range(self.min_n_components, self.max_n_components + 1)]
        for model_info in models_info:
            try:
                model = self.create_and_fit(model_info.num_components, self.X, self.lengths)
                if not model:
                    continue
                assert isinstance(model, GaussianHMM)
                logL = model.score(self.X, self.lengths)
                other_words_scores = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        try:
                            another_word_score = model.score(X, lengths)
                            other_words_scores.append(another_word_score)
                        except Exception as e:
                            if self.verbose:
                                print("Exception during traversing other words in DIC: {}, {}"
                                      .format(model_info.num_components, e))

                            continue

                m = len(other_words_scores)
                dic_score = logL - sum(other_words_scores) / (m - 1)
                model_info.score = dic_score
            except:
                if self.verbose:
                    print("Exception during calculation DIC score for n_components: {}, {}"
                          .format(model_info.num_components, e))

        return self.select_info_and_produce_model(models_info)


class ModelInfo:
    def __init__(self, num_components):
        self.num_components = num_components
        self.scores = None
        self.score = None


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        if len(self.sequences) <= 2:
            # if we have very few train data - smaller better to reduce overfitting.
            # But it's not fine situation for the further processing in any case
            return self.base_model(self.min_n_components)

        models_info = [ModelInfo(num_components)
                       for num_components in range(self.min_n_components, self.max_n_components + 1)]
        for model_info in models_info:
            model_info.scores = []

        split_method = KFold()

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            sequences_train, sequences_lengths_train = combine_sequences(cv_train_idx, self.sequences)
            sequences_test, sequences_lengths_test = combine_sequences(cv_test_idx, self.sequences)
            for model_info in models_info:
                score = self.fit_and_score(model_info.num_components,
                                           sequences_train, sequences_lengths_train,
                                           sequences_test, sequences_lengths_test)
                if score:
                    model_info.scores.append(score)
                else:
                    if self.verbose:
                        print("Score is None, training failed")

        for info in models_info:
            if info:
                if len(info.scores) > 0:
                    info.score = np.mean(info.scores)
                else:
                    info.score = float("-inf")

        return self.select_info_and_produce_model(models_info)

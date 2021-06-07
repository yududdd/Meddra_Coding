__author__ = "Yu Du"
__Email__ = "yu.du@clinchoice.com"
__date__ = "Dec 12,2020"

########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


class embedding:
    def __init__(self, terms, medra):
        self.terms = terms
        self.data_vocab, index = {}, 1
        self.data_vocab['<pad>'] = 0
        self.medra_dict, m_index = {}, 1
        self.medra_dict['<pad>'] = 0

        for i, v in terms.items():
            for word in v.split():
                if word not in self.data_vocab:
                    self.data_vocab[word] = index
                    index += 1

        for i, v in medra['pt_name'].items():
            for word in v.split():
                if word not in self.medra_dict:
                    self.medra_dict[word] = m_index
                    m_index += 1
        # Create reverse dictionary to map index back to words
        self.inverse_datavocab = {index: token for token, index in self.data_vocab.items()}
        self.inverse_medradict = {index: token for token, index in self.medra_dict.items()}
        self.model = None

    def train_embedding(self, **kargs):
        """
        Train the Word2Vec embedding using the Medra dictionary.
        :param args: passing any number of arguments, please refer to the word 2 vec documentation
        :return:the model created
        """
        self.model = Word2Vec([self.terms], min_count=1, size=50, window=3, workers=4, iter=100)
        return self.model

    @staticmethod
    def tsne_plot(model):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = []

        for word in model.wv.vocab:
            tokens.append(model[word])
            labels.append(word)

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

    @staticmethod
    def cosine_similarity(u, v):
        """
        Cosine similarity reflects the degree of similariy between u and v

        Arguments:
            u -- a word vector of shape (n,)
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """
        distance = 0.0
        # Compute the dot product between u and v (≈1 line)
        dot = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        # Compute the L2 norm of v (≈1 line)
        norm_v = np.linalg.norm(v)
        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = dot / (norm_u * norm_v)
        return cosine_similarity


if __name__ == "__main__":
    print("embedding class for auto encoder project...")
else:
    pass

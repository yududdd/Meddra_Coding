__author__ = "Yu Du"
__Email__ = "yu.du@clinchoice.com"
__date__ = "Dec 12,2020"

########################################################################################################################
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    """
    General class for preprocessing the input file including the Medra dictionary and raw data from EDC system
    """
    def __init__(self, raw_data, medra):
        """
        This class is designed to combine the preform preprocess of the raw datasets with focus on string
        preprocessing
        :param raw_data: the raw data wish to be go through pre-process
        :param dict: the medra dictionary
        """
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('wordnet')
        except LookupError:
            # If it does not exist, the program downloads the stopwords.
            nltk.download('stopwords', quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download('stopwords', download_dir='nltk_packages', quiet=True)
        self.raw_data = raw_data
        self.medra = medra

    def select_version(self, version):
        """
        Select particular data dictionary version to work on
        :param version: the version of raw data want to work on
        """
        try:
            self.raw_data['Version'] = pd.to_numeric(self.raw_data['Version'])
            self.raw_data = self.raw_data.loc[self.raw_data['Version'] == version]
            self.raw_data.drop("Version", axis=1, inplace=True) # Drop the version after use
        except KeyError:
            print("Select version is only applicable to raw datasets")

    def drop_row(self, column, length):
        """
        Drop the entire row when data in specified column greater than the length
        :param data: data to perform the action
        :param
        """
        self.medra.drop(self.medra[column][self.medra[column].apply(lambda x: len(x.split(" ")) > length)].index, inplace=True)


    def string_processor(self, x, grammer):
        """
        Method to preprocess the string, includes following process:
        1. lower case
        2. remove punctuation
        3. remove stop words
        4. stem or lemmatize the word: i.e. for grammatical reasons, d documents are going to use different forms of a
        word, such as organize, organizes, and organizing.
        For the difference between lemmatization and stemming,
        https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/
        :param grammer: "stem" or "lemma"
        :return: return a cleaned version of string (particularly the term in raw datasets, i.e. Verbatim Term in AE)
        """
        sw = stopwords.words('english')
        # Stemming
        stemmer = SnowballStemmer("english")
        # lemmatization
        lemma = WordNetLemmatizer()

        if grammer == 'stem':
            x_cln = ' '.join([stemmer.stem(i) for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split() if i not in sw]).lower()
        elif grammer == 'lemma':
            x_cln = ' '.join([lemma.lemmatize(i) for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split() if i not in sw]).lower()
        elif grammer == "medra":
            # x_cln = ' '.join([i.strip() for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split() if i not in sw]).lower() # keep the hyphen and numbers for the medra dictionary
            x_cln = ' '.join([i.strip() for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split() if i not in sw]).lower()
        else:
            # x_cln = ' '.join([i.strip() for i in re.sub(r'[^\w\s]+',' ', x).split() if i not in sw]).lower()
            x_cln = ' '.join([i.strip() for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split() if i not in sw]).lower()
        return x_cln


    def pipe_line(self):
        ################################################ Medra Preprocess ##################################################################################################
        # self.drop_row("llt_name", 15) # drop len greater than 15
        self.medra.loc[:,'llt_name'] = self.medra.loc[:,'llt_name'].apply(lambda x: self.string_processor(x, "medra")) # lower the case and remove punctuation
        self.medra.loc[:, 'pt_name'] = self.medra.loc[:, 'pt_name'].apply(lambda x: self.string_processor(x, "medra")) # lower the case and remove punctuation
        self.medra.columns =  ['LLT', 'DECOD', 'SOC']
        self.medra.drop_duplicates(inplace=True)
        self.medra.reset_index(inplace=True,drop=True) # reset the index
        ################################################# Raw Preprocess ###################################################################################################
        # self.select_version(22) # select version 22 for raw data
        # self.raw_data.drop("version", axis=1, inplace=True) #Drop the version column
        # self.raw_data.drop("AEBODSYS", axis=1, inplace=True) # Currently ignore the AEBODSYS
        self.raw_data.loc[:, "Verbatim Term"] = self.raw_data.loc[:, "Verbatim Term"].apply(lambda x: self.string_processor(str(x), "lemma")) # do the lemmatization for the raw term
        self.raw_data.loc[:, "LLT Name"] = self.raw_data.loc[:, "LLT Name"].apply(lambda x: self.string_processor(str(x), None)) # do the lemmatization for the raw term
        # self.raw_data.loc[:, "AEDECOD"] = self.raw_data.loc[:, "AEDECOD"].apply(lambda x: self.string_processor(x, None)) # Only lower the cases for the AEDECODE
        self.raw_data.drop("Version", axis=1, inplace=True) #Drop the version column
        # self.raw_data.drop_duplicates(subset=['Verbatim Term','LLT Name'], keep='first', inplace=True) # please note need to drop the duplicates and only left with unique datasets to give each sample same weights
        # self.raw_data.replace('', np.NaN, inplace=True)
        self.raw_data.dropna(inplace=True, how='any') # drop na in case if any
        ls = ['drug-induced hypothyroidism', 'limp hair', 'mobitz type', 'unilateral glaucoma', 'taste', 'fibrin dimer increased', 'vitamin increased',
        'disorders esophagus', 'electrocardiogram wave amplitude decreased', 'electrocardiogram wave inversion', 'immune-mediated hypothyroidism',
        'splenic artery embolization', 'hand eczema', 'skin pain', 'nan', 'iodine contrast media allergy', 'lymphoid follicle hyperplasia',
        'heart failure nyha class', 'vitamin decreased', 'knee replacement', 'wave flattening', 'negative wave', 'medication-related osteonecrosis jaw',
        'inverted waves', 'blood 1 25-dihydroxy vitamin increased', 'immunoglobulin decreased', 'wave inversion', 'oesophageal refflux', 'acute otitis externa',
         'disorders intestine', 'lower anterior resection', 'hypocortisolism', 'adenocarcinoma prostate stage', 'electrocardiogram wave abnormal', 'herpes simplex type',
          'post procedural erythema', 'blood 1 25-dihydroxy vitamin decreased', 'influenza virus infection', 'blood 1 25-dihydroxy vitamin increased']
        self.raw_data = self.raw_data[~self.raw_data["LLT Name"].isin(ls)] # we drop these AEDECOD in becase these PT are NOT FOUND in medra V22
        self.raw_data.columns =  ['TERM', 'LLT']
        self.raw_data.reset_index(inplace=True,drop=True) # reset the index
        return self.medra, self.raw_data


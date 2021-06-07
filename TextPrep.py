__author__ = "Yu Du, Leslie Xia, and Emma Li"
__Email__ = "Yu Du <yu.du@clinchoice.com>, Emma Li <emma.li@clinchoice.com>, Leslie Xia <leslie.xia@clinchoice.com>"
__date__ = "Apr 13,2020"
__version__  = "0.2.0"
__status__ = "UAT"
########################################################################################################################
#			              				      System Argument Usage													   #
# arg[1]: Input file pending for manual coding										    							   #
# arg[2]: Meddra dictionary version should be used																	   #
# python ./TextPrep.py AE_coding_MedDRA_v23.0_Auto-coder_Test_file.xlsx 22											   #
########################################################################################################################


import warnings
warnings.filterwarnings('ignore')
import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sas7bdat import SAS7BDAT
from tqdm import tqdm
import re
import pickle

from tensorflow import keras

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from greedy_search import *
from utils import *

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    #If it does not exist, the program downloads the stopwords.
    nltk.download('stopwords', quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download('stopwords', download_dir='nltk_packages', quiet=True)



class decoder(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.PATH = './data/'
        self.MEDDRA_PATH = self.PATH + 'meddra/'

        self.meddra = dict()
        self.rawdata = None
        self.rawdata_cln = None
        self.meddra_cln = None

        self.data_mapper = None # combine raw term and term after cleaning, so we can map back to the var
        self.meddra_mapper = None # similar function as above



    def get_meddra(self,ver_n):
        """
        Intro:Read in input data and dictionary data by automatically detecting file type (either sas7bdat
              or xlsx format);
        External functions: None
        Parameter:Select dictionary version by inputting numeric dictionary version parameter: ver_n
        Return:medra (dictionary data:llt,pt,etc.)
                ae (raw ae test data).
        """
        all_meddra = os.listdir(self.MEDDRA_PATH)
        all_data = os.listdir(self.PATH)

        for j in all_meddra:
            root = os.path.splitext(j)[0]
            name = os.path.splitext(j)[0]+os.path.splitext(j)[1]
            version = root.split("_")[2][1:]
            # print(root)

            if os.path.splitext(j)[1] == '.xlsx':
                self.meddra[int(version)] = pd.read_excel(self.MEDDRA_PATH+name)
            elif os.path.splitext(j)[1] == '.sas7bdat':
                self.meddra[int(version)] =SAS7BDAT(self.MEDDRA_PATH+name,encoding='gb2312').to_data_frame()
        try:
            print('- Read in Meddra Dictionary data version',ver_n,'The dataset contains ',len(self.meddra[ver_n]),' records')
        except KeyError:
            print('- No version found. Please verify the selected version of Meddra dictionary is loaded.')
        return self.meddra[ver_n]

    def get_rawdata(self):
        in_file=pd.read_excel(self.PATH + self.file_name)
        	#, sheet_name='Manual Encoding')
        print('- Read in Input data file. The dataset contains ',len(in_file),' records')
        in_file=in_file[['Verbatim Term']]
        in_file.rename(columns={'Verbatim Term':'AETERM'},inplace=True)
        try:
            ae=in_file.loc[in_file['version']==str(ver_n)]
        except:
            ae=in_file
        return in_file

    def load(self, ver):
        self.rawdata = self.get_rawdata()
        self.meddra = self.get_meddra(ver)
        return self.rawdata, self.meddra


    def match(self, right):
        matched = self.rawdata.reset_index().merge(self.meddra, how='inner',left_on="AETERM", right_on=right).set_index('index')
        unmatched = self.rawdata.reset_index().merge(self.meddra, how='outer' ,indicator=True, left_on="AETERM", right_on=right).loc[lambda x : x['_merge']=='left_only'].set_index('index')
        print("-There are ",len(matched),",",round((len(matched)/len(self.rawdata))*100, 2), "% records can be exactly matched")
        print("-There are ",len(unmatched),",",round((len(unmatched)/len(self.rawdata))*100, 2),"% records can't be directly matched")
        matched['confidence'] = str(100) + '%'
        matched['method'] = 'Exact Match'
        return matched, unmatched[['AETERM']]


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
        :return: return a cleaned version of string (particularly the term in raw datasets, i.e. AETERM in AE)
        """
        sw = stopwords.words('english')
        # Stemming
        stemmer = SnowballStemmer("english")
        # lemmatization
        lemma = WordNetLemmatizer()

        if grammer == 'stem':
            x_cln = ' '.join([stemmer.stem(i) for i in re.sub(r'[^a-zA-Z0-9-]',' ', str(x)).split() if i not in sw]).lower()
        elif grammer == 'lemma':
            x_cln = ' '.join([lemma.lemmatize(i) for i in re.sub(r'[^a-zA-Z0-9-]',' ', str(x)).split() if i not in sw]).lower()
        else:
            # x_cln = ' '.join([i.strip() for i in re.sub(r'[^\w\s]+',' ', x).split() if i not in sw]).lower()
            x_cln = ' '.join([i.strip() for i in re.sub(r'[^a-zA-Z0-9-]',' ', str(x)).split() if i not in sw]).lower()
        return x_cln

    def process(self, grammer, not_match):
        data_cln = pd.DataFrame()
        data_cln['AETERM']  = not_match['AETERM'].apply(lambda x: self.string_processor(x, grammer))
        meddra_cln = pd.DataFrame()
        meddra_cln['llt_name'] = self.meddra['llt_name'].apply(lambda x: self.string_processor(x, None))
        meddra_cln['pt_name'] = self.meddra['pt_name'].apply(lambda x: self.string_processor(x, None))
        meddra_cln.rename({'llt_name':'llt_name_cln', 'pt_name':'pt_name_cln'}, axis=1, inplace=True)# change the column header for cln_meddra
        print("-Preprocess successfully using grammer:", grammer)
        data_cln_new = data_cln.rename(columns={'AETERM':"AETERM_cln"})
        self.data_mapper = pd.concat([self.rawdata, data_cln_new], axis=1) #create a helper frame to map back to the raw term
        self.meddra_mapper = pd.concat([self.meddra, meddra_cln], axis=1) #create a helper frame to map back to the raw meddra
        return data_cln, meddra_cln


    def greedy_search(self,not_match):
        """
        Method to implement greedy search algo
        :param not_match: cleaned version of verbatim aeterm that is not matched by "excat"
        :return: 2 dataframe
        1: all_data: mapped by greedy search method, columns: AETERM(raw), greedy_prediction(LLT in meddra), method, confidence
        2. data_to_model: AETERM, AETERM_cln
        """
        meddra22_split = pickle.load(open('./data/pickle/meddra22_split.pkl', 'rb'))
        # three main methods:
        # 1. searchMatchLLTs()
        # 2. GivenLLTConfidence()
        # 3. recommendFromGreedy()
        rec_LLT_orig, rec_LLT_count = searchMatchLLTs(indata=not_match, dict=self.meddra_cln, split_meddra=meddra22_split)
        not_match['rec_LLT_orig'], not_match['rec_LLT_count'] = rec_LLT_orig, rec_LLT_count
        top_LLTs, rec_LLT_prob, diff_2nd_llt = GiveLLTConfidence(llt_dicts=rec_LLT_count)
        not_match['top_LLTs'], not_match['rec_LLT_prob'], not_match['diff_2nd_llt'] = top_LLTs, rec_LLT_prob, diff_2nd_llt
        meddra22_clnllt_to_llt = pickle.load(open('./data/pickle/meddra22_clnllt_to_llt.pkl', 'rb'))
        greedy_out_match, data_to_model = recommendFromGreedy(threshold=0.2, pred_data=not_match, llt_mapper=meddra22_clnllt_to_llt, highest_LLTs=top_LLTs, LLT_probs=rec_LLT_prob, diff_2nd_llt=diff_2nd_llt)
        print("-There are ",len(greedy_out_match),",",round((len(greedy_out_match)/len(self.rawdata))*100, 2), "% records recommended by greedy search")
        print("-There are ",len(data_to_model),",",round((len(data_to_model)/len(self.rawdata))*100, 2),"% records pass to machine learning model")
        all_data = pd.merge(greedy_out_match, self.data_mapper, left_index=True, right_index=True)
        data_to_model = pd.merge(data_to_model, self.data_mapper, left_index=True, right_index=True)
        data_to_model = data_to_model[['AETERM_y', 'AETERM_cln']]
        data_to_model.rename(columns={'AETERM_y':'AETERM'}, inplace=True)
        all_data = all_data[['AETERM_y','greedy_prediction', 'confidence']]
        all_data.rename(columns={'AETERM_y':'AETERM'}, inplace=True)
        all_data = all_data.reset_index().merge(self.meddra,how="inner",left_on="greedy_prediction",right_on='llt_name').set_index('index')
        all_data.drop(['greedy_prediction'],axis=1,inplace=True)
        all_data['method'] = 'Greedy Search'

        return all_data, data_to_model

    def nlp_predict(self, data_to_model):
        """
        Method to implement machine learning pred algo
        :param data_to_model: passed from greedy search algo: 2 columns: AETERM(raw), AETERM_cln
        :return: machine learning predicted output
        AETERM(raw): because input dataframe contains both raw and clean, use clean to pass ml and use raw to generate output
        Mapped all meddra columns: includes llt, pt, soc, hlt etc.
        """
        WINDOWS_Size = 9
        X_testls_new = [w.split() for w in data_to_model['AETERM_cln']] # use the clean terms to make the prediction
        word_to_index, index_to_word, word_to_vec_map = read_emb_vecs('./data/embeddings/vocab.tsv', './data/embeddings/vectors.tsv')
        Xtest_new=emdlayer(WINDOWS_Size, X_testls_new, 400, word_to_vec_map)
        model = keras.models.load_model('./data/model/model_h1.LLT.M22.15_Apr_21')
        print(model.summary())
        y_p = model.predict(Xtest_new)
        decoder =  pickle.load(open('./data/pickle/lltdecoder.pkl', 'rb'))
        y_pred = [decoder[i] if i in decoder.keys() else "NA" for i in y_p.argmax(axis=1)] # This is the predicted label
        print("-There are ",len(y_pred),"records predicted by machine learning")
        data = {"Verbatim Term": data_to_model['AETERM'], "Predict LLT": y_pred, "method": "Machine Learning", 'confidence':'~30%'}
        s0 = pd.DataFrame(data)
        comb_meddra = s0.reset_index().merge(self.meddra_mapper, left_on= "Predict LLT", right_on='llt_name_cln', how = 'left').set_index('index') # here we map the predict label to meddra to get all
        # since orignal dataset may contain duplicates, after the left/inner join of the predicted dataset, there will be duplicate index, this line remove duplicate index and keep the first. 
        comb_meddra = comb_meddra[~comb_meddra.index.duplicated(keep='first')]
        comb_meddra.drop(['Predict LLT','llt_name_cln', 'pt_name_cln'], axis=1, inplace=True)
        out = pd.DataFrame(comb_meddra)
        return out

    def out(self,exact_match, greedy_out_match, ml_pred):
        exact_match.rename(columns={'AETERM':'Verbatim Term'},inplace=True)
        greedy_out_match.rename(columns={'AETERM':'Verbatim Term'},inplace=True)
        frames = [exact_match, greedy_out_match, ml_pred]
        out = pd.concat(frames, ignore_index=False)
        out = out.sort_index()
        out.to_csv('./output/output.csv', header=True)


    def run(self, version):
        """
        Main method to run the pipeline
        The output is generated to ./output/
        The log text is generated to ./ , and used to save the command-line output.
        :param version: command line argument indicate the meddra version to be used.
        """
        # datetime object containing current date and time
        startTime= datetime.now()
        # mm/dd/YY H:M:S
        dt_string = startTime.strftime("%m/%d/%Y %H:%M:%S")
        print()
        print("Program Compiled Date and Time: ", dt_string, '\n')


        print('Step 1 of 6: Importing data...\n')
        self.load(version)

        print()
        print('Step 2 of 6: Exact matching...\n')
        exact_match, not_match = self.match('llt_name')


        print()
        print('Step 3 of 6: Data Processing...\n')
        not_match_cln, self.meddra_cln = self.process('stem', not_match)


        print()
        print('Step 4 of 6: Implementing Similarity Matching...')
        print()
        greedy_out_match, data_to_model = self.greedy_search(not_match_cln)


        print()
        print('Step 5 of 6: Implementing Machine Learning Prediction...')
        ml_pred = self.nlp_predict(data_to_model)


        print()
        print('Step 6 of 6: Generating output...')
        self.out(exact_match, greedy_out_match, ml_pred)

        print()
        endTime= datetime.now()
        dt_string2 = endTime.strftime("%m/%d/%Y %H:%M:%S")
        print("Program Finished Date and Time: ", dt_string2, '\n')
        print("Script Time Used: ", endTime - startTime)
        print()


if __name__ == "__main__":
    # sys.stdout = open("./log.txt", "w")
    print('############################# Run-Time Summary Log #############################')

    coder = decoder(sys.argv[1])
    coder.run(int(sys.argv[2]))

    print('############################## Program Finished #################################')
    # sys.stdout.close()

else:
    pass








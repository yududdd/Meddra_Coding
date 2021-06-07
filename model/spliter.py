__author__ = "Yu Du"
__Email__ = "yu.du@clinchoice.com"
__date__ = "Mar 03,2020"

########################################################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split


class Spliter:
    """
    General class for preprocessing the input file including the Medra dictionary and raw data from EDC system
    """

    def __init__(self, data, medra):
        """
        This class is intend to split all data into trian and test data.
        The train data will be contain all the medra data and the test data will contain only the raw data.
        :param raw_data: the raw data wish to be go through pre-process
        :param dict: the medra dictionary
        """
        self.data = data
        self.medra = medra
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_lst = None
        self.X_testlst = None

    def split(self, test_size=0.2):
        """
        concat the splited train with medra
        :param: the percentage of test size
        """
        X, y = self.data['TERM'], self.data['LLT']
        # rawX_train, self.X_test, rawy_train, self.y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.08, shuffle=True, random_state=42)
        # combine raw ae term to llt in the dictionary to expand training dataset
        # 4/5/2021 back to use only term for training and llt for testing
        # self.X_train = pd.concat([self.medra['LLT'], rawX_train])
        self.X_train.reset_index(drop=True, inplace=True)
        # self.y_train = pd.concat([self.medra['LLT'], rawy_train])
        self.y_train.reset_index(drop=True, inplace=True)
        self.X_lst =[w.split() for w in self.X_train]
        self.X_testlst =[w.split() for w in self.X_test]


    def get_train_test(self):
        """
        getter for getting splited data
        """
        self.split()
        return self.X_train, self.X_test, self.y_train, self.y_test, self.X_lst, self.X_testlst

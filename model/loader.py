__author__ = "Yu Du"
__Email__ = "yu.du@clinchoice.com"
__date__ = "Dec 12,2020"

########################################################################################################################
import pandas as pd
import re
import os
from sas7bdat import SAS7BDAT


class Loader:
    """
    General class for preprocessing the input file including the Medra dictionary and raw data from EDC
    """
    def __init__(self, dict_name):
        """
        This class is designed to combine the load and preform preprocess of the raw datasets with focus on string
        preprocessing
        Note: Loading the medra dictionary would take more 30s, please be patient.
        :param file_name: a list of file name of the input file, i.e. the AE, CM datasets
        """
        PATH = './'
        self.rawdf = pd.DataFrame()
        self.rawdf = pd.read_excel(PATH + "base.xlsx")
        # self.rawdf = pd.concat([self.rawdf, new])
        self.dict = pd.DataFrame()
        for file in dict_name:
            m_dict = pd.read_excel(PATH + file + '.xlsx', usecols='B,E, K') # column 'B, E' refers to the LLT, PT, and SOC Names
            self.dict = pd.concat([self.dict, m_dict])

    def __getattribute__(self, item):
        """
        Getter method to get attribute
        :param item: item want to get
        :return: the attribute via dot method
        """
        return super(Loader, self).__getattribute__(item)

    def get_version(self):
        return self.data.groupby(['version']).size()

    def select_version(self, version):
        """
        select specific medra version to work on
        :param version: medra version number
        :return:the raw datasets with specified version number
        """
        self.rawdf['version'] = pd.to_numeric(self.rawdf['version'])
        self.rawdf = self.rawdf.loc[self.rawdf['version'] == version]
        return self.rawdf


if __name__ == "__main__":
    # p = processor("ae.sas7bdat", "meddra_dict_v22")
    print("Data loaded!")
else:
    pass

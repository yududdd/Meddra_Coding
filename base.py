__author__ = "Yu Du, Leslie Xia"
__Email__ = "Yu Du <yu.du@clinchoice.com>, Leslie Xia <leslie.xia@clinchoice.com>"
__date__ = "May 25,2021"

########################################################################################################################
import pandas as pd
import numpy as np
import sys
import os
import re
from sas7bdat import SAS7BDAT


class base():
    def __init__(self):
        self.path= './data/addin/' #path of add-in dataset
        self.path_b='./data/base.xlsx' #Path of base dataset

    def read_base(self,path_b):
        base=pd.read_excel(self.path_b)
        return base

    def addin_data(self,path):
        dirs = os.listdir(self.path)
        data = pd.DataFrame()

        for file in dirs:
            tmp = None
            if os.path.splitext(file)[1] == '.sas7bdat':
                tmp = SAS7BDAT(self.path + file,encoding='gb2312').to_data_frame()
                print("- Load sas file " + file)
            elif os.path.splitext(file)[1] == '.xlsx':
                tmp = pd.read_excel(self.path + file, sheet_name=None)
                print("- Load excel file " + file)
            else:
                print("None acceptable file format " + file)

            if type(tmp) is dict:
                for value in tmp.values():
                    value.columns = value.columns.str.lower()
                    try:
                        value = value[['verbatim term', 'llt name', 'version']]
                    except:
                        sys.exit("The added file must contain three columns: verbatim term, llt name, version (case insensitive)")
                    data = pd.concat([data, value], ignore_index=True)
            else:
                if isinstance(tmp, pd.DataFrame):
                    tmp.columns = tmp.columns.str.lower()
                    try:
                        tmp = tmp[['verbatim term', 'llt name', 'version']]
                    except:
                        sys.exit("The added file, " + str() + ", must contain three columns: verbatim term, llt name, version (case insensitive)")
                    data = pd.concat([data, tmp], ignore_index=True)
        return data

    def string_processor(self, x):
        x_cln = ' '.join([i.strip() for i in re.sub(r'[^a-zA-Z0-9-]',' ', x).split()]).lower()
        return x_cln

    def process_addin(self):
        data = self.addin_data(self.path)
        data.columns= data.columns.str.lower()
        try:
            data = data[['verbatim term', 'llt name', 'version']]
        except:
            sys.exit("The added file must contain three columns: verbatim term, llt name, version (case insensitive)")
        # only turn into lower case and trim the empty spaces
        data['verbatim term'] = data['verbatim term'].apply(lambda x: self.string_processor(str(x)))
        data['llt name'] = data['llt name'].str.lower()
        data = data.rename(columns = {"verbatim term":"Verbatim Term", "llt name": "LLT Name", "version":"Version"})
        data = data.dropna(how='any')
        return data

    def concat_(self):
        base = self.read_base(self.path_b)
        print("The previous base dataset contains "+str(len(base))+ " observations")
        dirs = os.listdir(self.path)
        if len(dirs) == 0:
            return base
        else:
            new_data = self.process_addin()
            # new_data = new_data[['Verbatim Term', 'LLT Name', 'Version']
            base.reset_index(drop=True, inplace=True)
            new_data.reset_index(drop=True, inplace=True)
            print("new dataset(s) contain " + str(len(new_data)) + " observations")
            frames = [base,new_data]
            data_all= pd.concat(frames, axis=0, ignore_index=True)
            data_all_ndup=data_all.drop_duplicates(subset=['Verbatim Term'],keep='first')
            data_all_ndup.reset_index(drop=True, inplace=True)
            print("Adding completed! The new base dataset contains "+str(len(data_all_ndup))+ 
                " observations." )
            return data_all_ndup

if __name__ == "__main__":
    base = base()
    new_base = base.concat_()
    new_base.to_excel(base.path_b, header=True, index=False)
else:
    pass

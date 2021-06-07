__author__ = "Emma Li"
__Email__ = "emma.li@clinchoice.com"
__date__ = "March 20,2021"
######################################################################################################

import pandas as pd
import numpy as np
import re
import os
from statsmodels.distributions.empirical_distribution import ECDF

def searchMatchLLTs (indata, dict, split_meddra):
    '''
    1.
        For each cleaned record in input data, split cleaned Verbatim terms into phrase with length from 1 to 10.
        Store these phrase and cleaned LLTs containing these phrase into a dictionary.
        Thus, each record will have one corresponding dictionary. These dictionaries are wrapped in a list and returned.
    2.
        For each cleaned record in input data, that is, for each cleaned Verbatim Term, create a dictionary as a counter,
        with matched LLTs as key, weighted number of appearance of each LLT as value).
        Weight of appearance count is calculated as sum of the following results:
            1                            : each single word match,
            2 ** k                       : each phrase of length k match,
            2 ** 10 = 2048               : exactly matched LLT
            (2 ** 10 + 2 ** 11)/2 = 1536 : LLT is subset of Verbatim term
            (2 ** 10 + 2 ** 11)/2 = 1536 : Verbatim term is subset of LLT
                                            #contain exact same words (ignore order of words)
            (2 ** 9 + 2 ** 10)/2 = 768   : Length of overlap part between LLT and Verbatim term >= 75% length of the LLT
            (2 ** 9 + 2 ** 10)/2 = 768   : Length of overlap part between LLT and Verbatim term >= 75% length of the Verbatim term

        These dictionaries are wrapped in a list and returned.

        Args:
            indata: Input raw dataset with cleaned Verbatim Terms.
            dict: Meddra dictionary dataset with cleaned LLTs.
            split_meddra: A dictionary with all split phrase from all LLTs (and PTs) as keys,
                          and cleaned LLTs (PTs) containing these phrase as values.

        (
            rec_col_orig: A list returned. Contains dictionaries, one dictionary for one input record.
                          Each dictionary has split phrases from each cleaned input Verbatim Term as keys,
                          and cleaned LLTs (PTs) from assigned Meddra version containing these phrase as values.
            rec_LLTs: A list returned. Contains dictionaries, one dictionary for one input record.
                      Each dictionary has matched LLTs from assigned Meddra version as key,
                      and weighted appearance count of these LLTs as value.
        )
    '''

    rec_col_orig, rec_LLTs = [], []


    for cln_verbatim in indata['AETERM']:
        if cln_verbatim in dict.llt_name_cln:
            rec_col_orig.append({cln_verbatim:cln_verbatim})
            rec_llt.append({cln_verbatim:2048})

        else:
            rec_list, rec_each_llt = {}, {}
            verbatims = str(cln_verbatim).split(' ')

            for x in verbatims:
                rec_list.setdefault(x, set())
                if x in split_meddra:
                    rec_list[x] = split_meddra[x]
                    for llt in rec_list[x]:
                        if llt in rec_each_llt:
                            rec_each_llt[llt] += 1
                        else:
                            rec_each_llt[llt] = 1

            for k in range(2, 11):
                if len(verbatims) >= k:
                    for i in range(len(verbatims) - (k - 1)):
                        phrase = ' '.join(verbatims[i:i + k])
                        rec_list.setdefault(phrase, set())
                        if phrase in split_meddra:
                            rec_list[phrase] = split_meddra[phrase]
                            for llt in rec_list[phrase]:
                                if llt in rec_each_llt:
                                    rec_each_llt[llt] += 2**k
                                else:
                                    rec_each_llt[llt] = 2**k

                if len(verbatims) == k:
                    if cln_verbatim in split_meddra:
                        for llt in split_meddra[cln_verbatim]:
                            if set(str(llt).split(' ')).issubset(set(verbatims)):
                                rec_each_llt[llt] += 1536
                            if set(verbatims).issubset(set(str(llt).split(' '))):
                                rec_each_llt[llt] += 1536

            for llt in rec_each_llt:
                len_overlap = len(set(str(llt).split(' ')).intersection(set(verbatims)))
                len_llt = len(set(str(llt).split(' ')))
                len_verbatims = len(set(verbatims))
                if len_overlap / len_llt >= 0.75:
                    rec_each_llt[llt] += 768
                if len_overlap / len_verbatims >= 0.75:
                    rec_each_llt[llt] += 768


            rec_col_orig.append(rec_list)
            rec_LLTs.append(rec_each_llt)

    return rec_col_orig, rec_LLTs

#rec_LLT_orig, rec_LLT_count = searchMatchLLTs(indata=ae2, dict=dict22, split_meddra=meddra22_split)
#ae2['rec_LLT_orig'], ae2['rec_LLT_count'] = rec_LLT_orig, rec_LLT_count


def GiveLLTConfidence(llt_dicts):
    '''
        For each dictionary element in the previous returned list rec_PTs from searchMatchLLTs(),
        give recommendation of the top matched LLT(s) according to percent of weighted appearance.

        Args:
            llt_dicts: Previous returned list rec_LLTs from searchMatchLLTs()
            highest_LLTs: List of lists, each sublist contains the recommended LLTs with
                         largest weighted probability of appearance.
            LLT_probs: The weighted probability of appearance of the final recommended LLT.
            prob_diff_2nd: Weighted probability of final recommended LLT - 2nd large Weighted probability of all matched LLT candidates
    '''
    highest_LLTs, LLT_probs, prob_diff_2nd = [], [], []

    for dict_item in llt_dicts:
        if len(dict_item) >= 1:
            count_sum = float(sum(dict_item.values()))
            each_top_llt = [k for k, v in dict_item.items() if v == max(dict_item.values())]
            highest_LLTs.append(each_top_llt)

            top_LLT_prob = dict_item[each_top_llt[0]] / count_sum
            LLT_probs.append(top_LLT_prob)

            if len(each_top_llt) == 1 and len(dict_item) > 1:
                #print(sorted(dict_item.values(), reverse=True)[1])
                second_llt_prob = sorted(dict_item.values(), reverse=True)[1] / count_sum
            elif len(each_top_llt) > 1:
                second_llt_prob = 0
            prob_diff_2nd.append(top_LLT_prob - second_llt_prob)

        else:
            highest_LLTs.append('0')
            LLT_probs.append(float(0))
            prob_diff_2nd.append(float(0))

    return highest_LLTs, LLT_probs, prob_diff_2nd

#top_LLTs, rec_LLT_prob, diff_2nd_llt = GiveLLTConfidence(llt_dicts=rec_LLT_count)
#ae2['top_LLTs'], ae2['rec_LLT_prob'], ae2['diff_2nd_llt'] = top_LLTs, rec_LLT_prob, diff_2nd_llt


def recommendFromGreedy(threshold, pred_data, llt_mapper, highest_LLTs, LLT_probs, diff_2nd_llt):
    '''
        Select a threshold of probability equals to 0.2,
        Keep the recommendations with probability > 0.2 and pass rest of records to model part.

    '''

    ##
    top_LLTs_number = [len(x) for x in highest_LLTs]
    pred_data['top_LLTs_number'] = top_LLTs_number
    rec_1st_llt = [x[0] for x in highest_LLTs]
    pred_data['rec_1st_llt'] = rec_1st_llt
    top_llts_str = [', '.join(x) for x in highest_LLTs]
    pred_data['top_llts_str'] = top_llts_str
    orig_len = list(pred_data.apply(lambda x: len(x.rec_LLT_orig), axis=1))
    pred_data['orig_len'] = orig_len
    cln_term_length = [len(x.split(' ')) for x in pred_data['AETERM']]
    pred_data['cln_term_length'] = cln_term_length
    


    ##
    ecdf = ECDF(LLT_probs)
    confidence = []
    for i in range(len(LLT_probs)):
        if 0.65 <= max(ecdf(LLT_probs)[i], (LLT_probs[i] + diff_2nd_llt[i])) < 1:
            confidence.append(max(ecdf(LLT_probs)[i], (LLT_probs[i] + diff_2nd_llt[i])))
        elif 0.65 <= max(ecdf(LLT_probs)[i], LLT_probs[i]) < 1:
            confidence.append(max(ecdf(LLT_probs)[i], LLT_probs[i]))
        elif 0.65 <= ecdf(LLT_probs)[i] + LLT_probs[i] < 1:
            confidence.append(ecdf(LLT_probs)[i] + LLT_probs[i])
        elif max(ecdf(LLT_probs)[i], LLT_probs[i]) > 0.98:
            confidence.append(0.98)
        elif 0.65 <= ecdf(LLT_probs)[i] < 1:
            confidence.append(ecdf(LLT_probs)[i])
        elif 0.65 <= LLT_probs[i] < 1:
            confidence.append(LLT_probs[i])
        else:
            confidence.append(0.65)


    pred_data['confidence'] = confidence
    pred_data['confidence'] = pred_data['confidence'].map(lambda n: '{:,.2%}'.format(n))


    ##
    pred_data['flag'] = np.where((pred_data['cln_term_length']<5) & (pred_data['top_LLTs_number']==1) & (pred_data['orig_len']<10) & ( (pred_data['diff_2nd_llt'] <0.5) | (pred_data['diff_2nd_llt'] > 0.9)) & (pred_data['rec_LLT_prob'] > threshold), 1, 0)
    rec_top1st_gt_thr = pred_data.loc[pred_data['flag']==1]
    greedy_prediction = [llt_mapper[cln_llt] for cln_llt in rec_top1st_gt_thr['rec_1st_llt']]
    rec_top1st_gt_thr['greedy_prediction'] = greedy_prediction
    #print(rec_top1st_gt_thr.head())
    data_from_greedy = rec_top1st_gt_thr[['AETERM', 'greedy_prediction', 'confidence']]
    #print(data_from_greedy.head())
    rest_data = pred_data.loc[pred_data['flag']==0]
    data_to_model = rest_data[['AETERM']]

    return data_from_greedy, data_to_model

#greedy_out_data, data_to_model = recommendFromGreedy(threshold=0.2, pred_data=ae2, llt_mapper=meddra22_clnllt_to_llt, highest_LLTs=top_LLTs, LLT_probs=rec_LLT_prob, diff_2nd_llt=diff_2nd_llt)

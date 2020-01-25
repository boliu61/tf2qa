#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: bo_liu
"""

import pickle
import numpy as np
import natural_questions.eval_utils as util
import natural_questions.nq_eval as nq_eval

root_dir = '/Users/bo_liu/Documents/tf2_qa/'
pred_dir = '/Users/bo_liu/Documents/tf2_qa/pred_jsons/' 


eval_set = 'dev'
if eval_set=='tiny-dev':  nq_gold_dict = util.read_annotation_from_one_split(root_dir+'tiny-dev/nq-dev-sample.jsonl.gz')
elif eval_set=='dev00':   nq_gold_dict = util.read_annotation_from_one_split(root_dir+'dev/nq-dev-00.jsonl.gz')
elif eval_set=='dev':  
    nq_gold_dict = util.read_annotation_from_one_split(root_dir+'dev/nq-dev-00.jsonl.gz')
    for i in range(1,5): nq_gold_dict.update( util.read_annotation_from_one_split(root_dir+f'dev/nq-dev-0{i}.jsonl.gz'))    
else:                     raise



class ScoreSummary(object):    
  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None  




######## ensemble nbest

def ensemble_nbest_to_nq_pred_dict(nbest_lst, wts, yes_thr=None, no_thr=None):
  def sigmoid(x): return 1 / (1 + np.exp(-x))
    
  assert len(nbest_lst)==len(wts) #and len(wts)>1
  nq_pred_dict = {}
  for i in range(1,len(nbest_lst)): assert set(nbest_lst[0].keys()) == set(nbest_lst[i].keys())
  
  for example_id in nbest_lst[0].keys():
    d_long = {} # key: (start_tok,end_tok), value: prob
    d_short = {}
    yes_logits,no_logits = 0,0
    for i,nbest in enumerate(nbest_lst):
        if yes_thr: yes_logits += nbest[example_id][0].answer_type_logits[1] * wts[i]
        if no_thr: no_logits += nbest[example_id][0].answer_type_logits[2] * wts[i]
        long_seen = set()
        short_seen = set()
        for score_summary in nbest[example_id]:
          pred_lbl = score_summary.predicted_label
          long_ans =  pred_lbl['long_answer'] 
          short_ans = pred_lbl['short_answers'][0]
          long_key = long_ans['start_token'], long_ans['end_token']
          short_key = short_ans['start_token'], short_ans['end_token']          
          if long_key not in long_seen:
              long_seen.add(long_key)
              if long_key not in d_long: d_long[long_key] = sigmoid(pred_lbl['long_answer_score']) * wts[i]
              else:                      d_long[long_key]+= sigmoid(pred_lbl['long_answer_score']) * wts[i]
          if short_key not in short_seen:
              short_seen.add(short_key)
              if short_key not in d_short: d_short[short_key] = sigmoid(pred_lbl['short_answers_score']) * wts[i]
              else:                        d_short[short_key]+= sigmoid(pred_lbl['short_answers_score']) * wts[i]                  

    (start_tok_long,end_tok_long),prob_long = sorted(list(d_long.items()),key=lambda x:x[1],reverse=True)[0]
    (start_tok_short,end_tok_short),prob_short = sorted(list(d_short.items()),key=lambda x:x[1],reverse=True)[0]

    short_answer_span_list = [util.Span(-1,-1,start_tok_short,end_tok_short)]
    yes_no_answer='none'        
    if yes_thr and yes_logits > yes_thr:
        yes_no_answer,prob_short = 'yes', 999999
        short_answer_span_list = [util.Span(-1,-1,-1,-1)]
    elif no_thr and no_logits > no_thr:
        yes_no_answer,prob_short = 'no', 999999
        short_answer_span_list = [util.Span(-1,-1,-1,-1)]
        
        
    nq_pred_dict[int(example_id)] = util.NQLabel(
                        example_id=example_id,
                        long_answer_span = util.Span(-1, -1, start_tok_long, end_tok_long),
                        short_answer_span_list = short_answer_span_list,
                        yes_no_answer=yes_no_answer,
                        long_score=prob_long,
                        short_score=prob_short)   
  return nq_pred_dict  


def print_r_at_p_table(answer_stats,targets=[],thr_in=None):
  """Pretty prints the R@P table for default targets."""
  opt_result, pr_table = nq_eval.compute_pr_curves(
      answer_stats, targets=targets)
  f1, precision, recall, threshold = opt_result
  
  if thr_in: threshold = thr_in
  
  tp = sum([x[2] and x[3]>=threshold for x in answer_stats])
  true = sum([x[0] for x in answer_stats])
  pred = sum([x[1] and x[3]>=threshold for x in answer_stats ])    
  
  if not thr_in:
      print('Optimal threshold: {:.5}'.format(threshold))
      print(' F1     /  P      /  R')
      print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
      for target, recall, precision, row in pr_table:
        print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
            target, recall, precision, row))
  else:
      precision = nq_eval.safe_divide(tp, pred)
      recall = nq_eval.safe_divide(tp, true)
      f1 = nq_eval.safe_divide(2*precision*recall, precision+recall)      
      print('Input threshold: {:.5}'.format(threshold))
      print(' F1     /  P      /  R')
      print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))      
  
  return threshold,tp,true,pred,f1

def score_answers(gold_annotation_dict, pred_dict,  thr_long=None,thr_short=None,sort_by_id=False):
  """Scores all answers for all documents.
  Args:
    gold_annotation_dict: a dict from example id to list of NQLabels.
    pred_dict: a dict from example id to list of NQLabels.
    sort_by_id: if True, don't compute F1; if False, compute F1 and print
  Returns:
    long_answer_stats: List of scores for long answers.
    short_answer_stats: List of scores for short answers.
  """
  # gold_annotation_dict = nq_gold_dict
  # pred_dict = nq_pred_dict

  gold_id_set = set(gold_annotation_dict.keys())
  pred_id_set = set(pred_dict.keys())

  if gold_id_set.symmetric_difference(pred_id_set):
    raise ValueError('ERROR: the example ids in gold annotations and example '
                     'ids in the prediction are not equal.')

  long_answer_stats = []
  short_answer_stats = []

  for example_id in gold_id_set:
    gold = gold_annotation_dict[example_id]
    pred = pred_dict[example_id]

    if sort_by_id:
      long_answer_stats.append(list(nq_eval.score_long_answer(gold, pred))+[example_id])
      short_answer_stats.append(list(nq_eval.score_short_answer(gold, pred))+[example_id])
    else:
      long_answer_stats.append(nq_eval.score_long_answer(gold, pred))
      short_answer_stats.append(nq_eval.score_short_answer(gold, pred))

  # use the 'score' column, which is last
  long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
  short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

  if not sort_by_id:
      print('-' * 20)
      print('LONG ANSWER R@P TABLE:')
      thr_long,tp_long,true_long,pred_long,f1_long = print_r_at_p_table(long_answer_stats,thr_in=thr_long)
      print('-' * 20)
      print('SHORT ANSWER R@P TABLE:')
      thr_short,tp_short,true_short,pred_short,f1_short = print_r_at_p_table(short_answer_stats,thr_in=thr_short)
    
      precision = nq_eval.safe_divide(tp_long+tp_short, pred_long+pred_short)
      recall = nq_eval.safe_divide(tp_long+tp_short, true_long+true_short)
      f1 = nq_eval.safe_divide(2*precision*recall, precision+recall)

      print('-' * 20)      
      print(' F1     /  P      /  R')
      print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))       
    
      return long_answer_stats, short_answer_stats, thr_long, thr_short, f1_long,f1_short,f1
  return long_answer_stats, short_answer_stats


######### tune ensemble weights 


pkl_lst=[        
 'nbest_dict_dev_wwm_stride_192_neg_0.01_0.04-64-2.00E-05_model7000_256_tpu.pkl',     # L
 'nbest_dict_dev_wwm_fix_top_level_bug_max_contexts_200_0.01_0.04-64-4.00E-05_model9500_256_tpu.pkl', #O
 'nbest_dict_dev_wwm_cased_fix_top_level_bug_0.01_0.04-64-4.50E-05_model8500_256_tpu.pkl', #R2
        ]
nbest_lst = [pickle.load(open(root_dir+'nbest_pkl/'+nbest_file,'rb')) for nbest_file in pkl_lst]

_ = score_answers(nq_gold_dict, ensemble_nbest_to_nq_pred_dict(nbest_lst, [1,1.1,0.8]))


#--------------------
#LONG ANSWER R@P TABLE:
#Optimal threshold: 2.3972
# F1     /  P      /  R
# 71.62% /  71.77% /  71.46%
#--------------------
#SHORT ANSWER R@P TABLE:
#Optimal threshold: 2.895
# F1     /  P      /  R
# 58.50% /  70.07% /  50.20%
#--------------------
# F1     /  P      /  R
# 66.47% /  71.18% /  62.35%

 
######## tune yes/no thr

for thr in np.arange(4.5,6.5,0.5):
    print('\n')
    print(thr)
    _ = score_answers(nq_gold_dict, 
            ensemble_nbest_to_nq_pred_dict(nbest_lst, [1,1.1,0.8],yes_thr=thr,no_thr=None))

#5.5
#--------------------
#LONG ANSWER R@P TABLE:
#Optimal threshold: 2.3972
# F1     /  P      /  R
# 71.62% /  71.77% /  71.46%
#--------------------
#SHORT ANSWER R@P TABLE:
#Optimal threshold: 2.895
# F1     /  P      /  R
# 59.61% /  70.04% /  51.88%
#--------------------
# F1     /  P      /  R
# 66.87% /  71.15% /  63.07%
    
    
for thr in np.arange(4.5,6.5,0.5):
    print('\n')
    print(thr)
    _ = score_answers(nq_gold_dict, 
            ensemble_nbest_to_nq_pred_dict(nbest_lst, [1,1.1,0.8],yes_thr=5.5,no_thr=thr))

#5.5
#--------------------
#LONG ANSWER R@P TABLE:
#Optimal threshold: 2.3972
# F1     /  P      /  R
# 71.62% /  71.77% /  71.46%
#--------------------
#SHORT ANSWER R@P TABLE:
#Optimal threshold: 2.895
# F1     /  P      /  R
# 59.86% /  70.10% /  52.23%
#--------------------
# F1     /  P      /  R
# 66.96% /  71.17% /  63.22%
    
    
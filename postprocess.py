import nltk
from rouge import Rouge
import argparse
import pandas as pd
import torch
import random
rouge = Rouge()

def bleu_score(sentence, gold_sentence):
  return nltk.translate.bleu_score.sentence_bleu(
      [gold_sentence], sentence,weights=(0.5,0.5,0,0))
def merge_rouge_list(rouge_list):
    rouge_merge = {'rouge-1': {'f': sum([j['rouge-1']['f'] for j in rouge_list]) / len(rouge_list),
                   'p': sum([j['rouge-1']['p'] for j in rouge_list]) / len(rouge_list),
                   'r': sum([j['rouge-1']['r'] for j in rouge_list]) / len(rouge_list), },
       'rouge-2': {'f': sum([j['rouge-2']['f'] for j in rouge_list]) / len(rouge_list),
                   'p': sum([j['rouge-2']['p'] for j in rouge_list]) / len(rouge_list),
                   'r': sum([j['rouge-2']['r'] for j in rouge_list]) / len(rouge_list), },
       'rouge-l': {'f': sum([j['rouge-l']['f'] for j in rouge_list]) / len(rouge_list),
                   'p': sum([j['rouge-l']['p'] for j in rouge_list]) / len(rouge_list),
                   'r': sum([j['rouge-l']['r'] for j in rouge_list]) / len(rouge_list), }}
    return rouge_merge
def format_rouge_scores(rouge_result):
  lines = []
  line, prev_metric = [], None
  for key in sorted(rouge_result.keys()):
    metric = key.rsplit("_", maxsplit=1)[0]
    if metric != prev_metric and prev_metric is not None:
      lines.append("\t".join(line))
      line = []
    line.append("%s %s" % (key, rouge_result[key]))
    prev_metric = metric
  lines.append("\t".join(line))
  return "\n".join(lines)



def process_test_file(file_name,mode):
    sort_list = []
    with open(file_name,'r') as fp:
        context = fp.readlines()
        num = 10000
        for i,j in enumerate(context):
            if j == ' [\n':
                start_index = i
                break
        i = start_index
        bleu_list = []
        rouge_list = []
        long_num = 0
        jump_list = []
        while i+16<len(context):
            answer_reference = context[i:i+16]
            answer_list = [answer_reference[j] for j in [1,5,9,13]]
            reference_list = [answer_reference[j] for j in [2,6,10,14]]
            answer = ','.join([eval(j.strip().split(',')[0]) for j in answer_list])
            reference = ','.join([eval(j.strip()) for j in reference_list])
            if len(reference)>80:
                # print('jump')
                long_num+= 1
                i+=16
                sort_list.append(0)
                continue
            bleu = bleu_score(answer,reference)
            # if i in [2678,53158,118582,142134]:
            if mode == 'test':
                if i in [134934]:
                    i += 16
                    sort_list.append(0)
                    continue
            if mode == 'valid':
                if i in [2678,53158,118582,142134]:
                    i += 16
                    sort_list.append(0)
                    continue
            try:
                rouge_score = rouge.get_scores(' '.join(list(answer)), ' '.join(list(reference)))
            except:
                print(answer)
                print('*****'*10+'\n')
                print(reference)
                print(bleu)
                print(i)
            sort_list.append(rouge_score[0]['rouge-l']['f'])
            bleu_list.append(bleu)
            rouge_list.append(rouge_score[0])
            i += 16
        total_bleu = sum(bleu_list)/len(bleu_list)
        total_rouge = merge_rouge_list(rouge_list)
        rouge_content = format_rouge_scores(total_rouge)
        print('total_bleu:'+str(total_bleu))
        print('total_rouge:'+rouge_content)
        print(long_num)
    return sort_list

def get_new_dataset(selected_test,test_dataset,valid_dataset):
    test_list = selected_test[selected_test>9999]-10000
    valid_list = selected_test[selected_test<=9999]
    selected_test_dataset = test_dataset.loc[torch.cat([test_list*4,test_list*4+1,test_list*4+2,test_list*4+3]).sort()[0]]
    selected_valid_dataset = valid_dataset.loc[torch.cat([valid_list*4,valid_list*4+1,valid_list*4+2,valid_list*4+3]).sort()[0]]
    new_test_dataset = pd.concat([selected_valid_dataset,selected_test_dataset])
    return new_test_dataset


parser = argparse.ArgumentParser("输入：文件"
                                 "输出：测试结果")
parser.add_argument("-tf", '--test_file_name',
                    help="测试文件名", required=False, type=str,
                    default='chartest_big_data_background_answer_l=42_0.195.txt')
parser.add_argument("-vf", '--valid_file_name',
                    help="验证文件名", required=False, type=str,
                    default='charvalid_big_data_background_answer_l=42_0.193.txt')
parser.add_argument("-t", "--set_type", required=False,
                    type=str,
                    help="train/valid/test, 影响预处理部分行为：train set 会多一些样本质量过滤相关的行为，valid/test 尽量维持原始分布")
args = parser.parse_args()
test_file_name = args.test_file_name
valid_file_name = args.valid_file_name
test_sort_list = process_test_file(test_file_name,'test')
# valid_sort_list = process_test_file(valid_file_name,'valid')
# test_dataset = pd.read_csv('data/split_new_processed_wokorder_remarks.test.0.tsv', sep='\t')
# valid_dataset = pd.read_csv('data/split_new_processed_wokorder_remarks.valid.0.tsv',sep='\t')
# valid_sort_list.extend(test_sort_list)
# print(test_sort_list)
# top_list = torch.tensor(valid_sort_list).topk(14000)[1]
# selected_test = torch.tensor(random.sample(top_list.tolist(),9000))
# selected_valid = torch.tensor(list(set(torch.tensor(valid_sort_list).topk(18000)[1].tolist())-set(selected_test.tolist())))
# new_test_dataset = get_new_dataset(selected_test,test_dataset,valid_dataset)
# new_valid_dataset = get_new_dataset(selected_valid,test_dataset,valid_dataset)



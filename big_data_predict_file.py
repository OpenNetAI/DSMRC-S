
import pandas as pd
from sklearn import metrics
from transformers import AutoTokenizer, BertForTokenClassification, AdamW
# from train_file import DialogueDataset
from transformers import TrainingArguments, Trainer
import torch
import numpy as np
import nltk
from torch.utils.data import Dataset
import json
import re
import os
import jieba
from rouge import Rouge
import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

def bleu_score(sentence, gold_sentence):
  return nltk.translate.bleu_score.sentence_bleu(
      [gold_sentence], sentence,weights=(0.5,0.5,0,0))

class DialogueDataset(Dataset):
    def __init__(self,path_to_file):
        self.dataset = pd.read_csv(path_to_file,sep='\t')
        self.question_dict = {0:'对话的背景是什么',1:'来电的诉求是什么',2:'问题的关键是什么',3:'解决方案是什么'}
        # self.get_labels()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        #answer_type = {0:background,1:purpose,2:key,3:solved_method}
        #question_dict = {0:'对话的背景是什么',1:'来电的诉求是什么',2:'问题的关键是什么',3:'解决的方案是什么'}
        row = self.dataset.loc[idx]
        dialogue = '&'.join([i['text'] for i in eval(row['asr_result'])])[:501]
        question = self.question_dict[row['answer_type']]
        label_list = [0]*10+eval(row['token_label']) +[0]*512
        label = torch.tensor(label_list[:512])
        tokenized_sample = tokenizer(question,dialogue,padding='max_length',max_length=512,return_tensors='pt')

        # dialogue = self.dataset.loc[idx, "dialogue"]
        # question_summary = self.dataset.loc[idx,'question_summary']
        # summary_label = torch.tensor(self.dataset.loc[idx,'background_label'])
        # tokenized_sample = tokenizer(question_summary,dialogue,padding='max_length',max_length=512,return_tensors='pt')
        tokenized_sample['input_ids'] = tokenized_sample['input_ids'].squeeze(0).to(device)
        tokenized_sample['token_type_ids'] = tokenized_sample['token_type_ids'].squeeze(0).to(device)
        tokenized_sample['attention_mask'] = tokenized_sample['attention_mask'].squeeze(0).to(device)
        tokenized_sample['labels'] = label.to(device)
        return tokenized_sample
    def get_labels(self):
        print('建立标签中...')
        background_label_list = []
        purpose_label_list = []
        for i, sample in self.dataset.iterrows():
            tokens = tokenizer(sample.question_summary,sample.dialogue)[0].tokens
            # l_len = len(tokens)
            # background_label = [0] * l_len# for [CLS] [SEP] [SEP]
            # purpose_label = [0] * l_len
            background_label = [0]*512
            purpose_label = [0]*512
            dialogue = sample.dialogue
            background = sample.background
            purpose = sample.purpose
            if i%1000 == 0:
                print(i)
            for j,char in enumerate(tokens):
                if char in set(background):
                    background_label[j] = 1
                if char in set(purpose):
                    purpose_label[j] = 1
            background_label_list.append(background_label)
            purpose_label_list.append(purpose_label)
        self.dataset['background_label'] = pd.Series(background_label_list)
        self.dataset['purpose_label'] = pd.Series(purpose_label_list)


if device.type == 'cpu':
    local = True
else:
    local = False
if local:
    model = BertForTokenClassification.from_pretrained('trained_model/checkpoint-70000')
    tokenizer = AutoTokenizer.from_pretrained('trained_model/checkpoint-70000')
    toy_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.test.0.tsv')
    batch_size = 64
else:
    model = BertForTokenClassification.from_pretrained('bigdata_test-tokenclassification/checkpoint-70000')
    tokenizer = AutoTokenizer.from_pretrained("bigdata_test-tokenclassification/checkpoint-70000")
    train_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.train.0.tsv')
    valid_dialogue_dataset = DialogueDataset('../data/new_valid.tsv')
    test_dialogue_dataset = DialogueDataset('../data/new_test.tsv')


    # toy_dialogue_dataset = DialogueDataset('../data/dialogue_toy.tsv')
    # train_dialogue_dataset = DialogueDataset('../data/dialogue_train.tsv')
    # valid_dialogue_dataset = DialogueDataset('../data/dialogue_valid.tsv')
    # test_dialogue_dataset = DialogueDataset('../data/dialogue_test.tsv')
    batch_size = 96


def compute_metrics(pred):
    y_true = pred.label_ids.flatten()
    y_pred = pred.predictions.argmax(-1).flatten()
    # results = metric.compute(predictions=preds, references=labels)
    results = {}
    results['precision'] = metrics.precision_score(y_true, y_pred, average='binary')
    results['recall'] = metrics.recall_score(y_true, y_pred, average='binary')
    results['f1'] = metrics.f1_score(y_true, y_pred, average='binary')
    results['accuracy'] = metrics.accuracy_score(y_true,y_pred)
    return {
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
        "accuracy": results["accuracy"],
    }

args = TrainingArguments(
    f"big-data-test-{'tokenclassification'}",
    # overwrite_output_dir=True,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_steps=10000,
    save_total_limit=2,
    num_train_epochs=5,
    weight_decay=0.01,
)
if local:
    trainer = Trainer(
        model.to(device),
        args,
        train_dataset=toy_dialogue_dataset,
        eval_dataset=toy_dialogue_dataset,
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
else:
    trainer = Trainer(
        model.to(device),
        args,
        train_dataset=train_dialogue_dataset,
        eval_dataset=valid_dialogue_dataset,
        # data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

#get strategy from training dataset
# dataset = toy_dialogue_dataset

# get answer from output probability of token classification layer
def get_answer_from_strategy(output_probability,sample,tokenized_result,method=1,l=5,window_length=3,alpha=0.4,distribution=None,level='word'):
    #0: fixed length answer 1: flexible length answer 2:parameter length answer
    def get_index_list(input_length,l,special_token_index):
        result = []
        for i in range(len(special_token_index)-1):
            interval = [special_token_index[i],special_token_index[i+1]]
            for j in range(interval[0]+1,interval[1]-l):
                result.append(j)
        return result
    def get_segment_output_probability(tokenized_tokens,word_list,output_probability):
        segment_output_probability = []
        i = 0
        j = 0
        while i<len(tokenized_tokens) and j < len(word_list):
            if tokenized_tokens[i] in ['[CLS]','[SEP]','[&]']:
                segment_output_probability.append(output_probability[i])
                i += 1
                j += 1
                continue
            if word_list[j].isdigit() or bool(re.search('[a-z]',word_list[j])):
                tmp_output_p = [output_probability[i]]
                while i+1<len(tokenized_tokens) and tokenized_tokens[i+1].startswith('##'):
                    i += 1
                    tmp_output_p.append(output_probability[i])
                segment_output_probability.append(sum(tmp_output_p)/len(tmp_output_p))
                i += 1
                j += 1
                continue
            else:
                l = len(word_list[j])
                segment_output_probability.append(sum(output_probability[i:i+l])/l)
                j += 1
                i += l
        return segment_output_probability

    word_list = tokenized_result.tokens()
    input_length = int(tokenized_result.attention_mask.sum())
    if level == 'word':
        input_str = tokenizer.decode(tokenized_result.input_ids)[:input_length].replace(' ','')
        question = input_str.split('[SEP]')[0].split('[CLS]')[1]
        dialogue_list = input_str.split('[SEP]')[1].split('&')
        word_segment_dialogue_list = [jieba.lcut(i) for i in dialogue_list]
        word_segment_total_list = []
        for i,sentence in enumerate(word_segment_dialogue_list):
            if i == 0:
                word_segment_total_list.append('[CLS]')
                word_segment_total_list.append(question)
                word_segment_total_list.append('[SEP]')
            word_segment_total_list.extend(sentence)
            if i == len(word_segment_dialogue_list) -1 :
                word_segment_total_list.append('[SEP]')
            else:
                word_segment_total_list.append('&')
        word_list = word_segment_total_list
        segment_output_probability = get_segment_output_probability(tokenized_result.tokens(),word_list,output_probability)
        # try :
        #     for i in word_list:
        #         if i in ['[CLS]','&','[SEP]'] or bool(re.search('[a-z]',i)):
        #             segment_output_probability.append(output_probability[j])
        #             j += 1
        #         else:
        #             segment_output_probability.append(sum(output_probability[j:j+len(i)])/len(i))
        #             j += len(i)
        # except:
        #     with open('error.txt','w') as f:
        #         f.write(word_list)
        #     print(word_list)
        output_probability = torch.tensor(segment_output_probability)
    if method == 0: #fixed length answer
        if level == 'word':
            input_length = len(word_list)
        special_token_index = [i for i,j in enumerate(word_list) if j in ['[CLS]','&','[SEP]']]
        index_list = get_index_list(input_length,l,special_token_index)
        density_list = [output_probability[i:i+l].sum() for i in index_list]
        while not density_list and l>1:
            l = l-1
            index_list = get_index_list(input_length, l, special_token_index)
            density_list = [output_probability[i:i + l].sum() for i in index_list]
        # density_list = [output_probability[i:i+l].sum() for i in range(input_length-l)]
        try:
            answer_start = index_list[np.argmax(density_list)]
        except:
            print(tokenized_result.tokens())
            print(word_list)
            print(output_probability)
        answer_list = word_list[answer_start:answer_start+l]
        # answer_list = [i for i in answer_list if i not in ['&', '[CLS]', '[SEP]', '[PAD]']]
        # while '&' in answer_list or '[CLS]' in answer_list or '[SEP]' in answer_list or '[PAD]' in answer_list:
        #     density_list[answer_start] = 0
        #     answer_start = np.argmax(density_list)
        #     answer_list = tokenized_result.tokens()[answer_start:answer_start + l]
        return ''.join(answer_list)
    if method == 1: #fiexible length answer
        if level == 'word':
            input_length = len(word_list)
        max_density = 0
        for deta_l in range(l-window_length,l+window_length+1):
            special_token_index = [i for i, j in enumerate(word_list) if j in ['[CLS]', '&', '[SEP]']]
            index_list = get_index_list(input_length, deta_l, special_token_index)
            density_list = [(output_probability[i:i + deta_l].sum() / deta_l) *(deta_l**alpha) for i in index_list]
            # density_list = [output_probability[i: i + deta_l].sum()/deta_l for i in range(input_length - deta_l)]
            if not density_list:
                continue
            if max(density_list) > max_density:
                answer_start = index_list[np.argmax(density_list)]
                answer_list = word_list[answer_start:answer_start+deta_l]
                # answer_list = [i for i in answer_list if i not in ['&','[CLS]','[SEP]','[PAD]']]
                answer = ''.join(answer_list)
                max_density = max(density_list)
                # while '&' in answer_list or '[CLS]' in answer_list or '[SEP]' in answer_list or '[PAD]' in answer_list:
                #     density_list[answer_start] = 0
                #     if max(density_list)<max_density:
                #         break
                #     answer_start = np.argmax(density_list)
                #     answer_list = tokenized_result.tokens()[answer_start:answer_start + deta_l]
                # else:
                #     max_density = max(density_list)
                #     answer = ''.join(answer_list)
        return answer
    if method == 2:
        max_utterance_length = max([len(i['text']) for i in eval(sample.asr_result)])
        max_density = 0
        for deta_l in range(1,max_utterance_length+1):
            if deta_l<=0:
                continue
            special_token_index = [i for i, j in enumerate(word_list) if j in ['[CLS]', '&', '[SEP]']]
            index_list = get_index_list(input_length, deta_l, special_token_index)
            density_list = [(output_probability[i:i + deta_l].sum() / deta_l) *(deta_l**alpha) for i in index_list]
            if not density_list:
                continue
            # density_list = [(output_probability[i:i + deta_l].sum() / deta_l)*(deta_l**alpha) for i in range(input_length - deta_l+1)]
            if max(density_list) > max_density:
                answer_start = index_list[np.argmax(density_list)]
                answer_list = word_list[answer_start:answer_start + deta_l]
                answer = ''.join(answer_list)
                max_density = max(density_list)
                # while '&' in answer_list or '[CLS]' in answer_list or '[SEP]' in answer_list or '[PAD]' in answer_list:
                #     density_list[answer_start] = 0
                #     if max(density_list) < max_density:
                #         break
                #     answer_start = np.argmax(density_list)
                #     answer_list = tokenized_result.tokens()[answer_start:answer_start + deta_l]
                # else:
                #     max_density = max(density_list)
                #     answer = ''.join(answer_list)
        return answer
    if method == 3:
        pass

rouge = Rouge()
# sets = {'train':train_dialogue_dataset,'valid':valid_dialogue_dataset,'test':test_dialogue_dataset}
if local:
    sets = {'toy':toy_dialogue_dataset}
else:
    sets = {'test':test_dialogue_dataset}
over_length = 0
level = 'word'
error_list = []
start_time = time.time()
l = 4
for method in range(2,3):
    for tmp in sets:
        dataset = sets[tmp]
        result = trainer.predict(dataset)
        for level in ['char']:
            for alpha in [i/10.0 for i in range(11)]:
                bleu_dict = {0:[],1:[],2:[],3:[]}
                rouge_dict = {0:[],1:[],2:[],3:[]}
                # dataset.dataset = dataset.dataset.loc[:99,:]

                output_probability = torch.tensor(result.predictions).softmax(dim=-1)[:, :, 1]
                print(tmp+' start evaluate method %d'%method)
                bleu_list = []
                answer_list = []
                # rouge_1_list = []
                # rouge_2_list = []
                # rouge_l_list = []
                for i in range(len(dataset)):
                    sample = dataset.dataset.loc[i]
                    # if sample.answer > 20:
                    #     over_length += 1
                    #     print(over_length)
                    #     continue
                    if method == 2 and i%5000 == 1:
                        print('*****' * 5)
                        print(level+tmp + '_method_' + str(method) + ':')
                        print('bleu score: %f' % (sum(bleu_list) / len(bleu_list)))
                        # print('rouge_1 score: %f' % (sum(rouge_1_list) / len(rouge_1_list)))
                        # print('rouge_2 score: %f' % (sum(rouge_2_list) / len(rouge_2_list)))
                        # print('rouge_l score: %f' % (sum(rouge_l_list) / len(rouge_l_list)))
                        print(i)
                    # l_dict = {0:12,1:6,2:13,3:13}
                    # l = l_dict[sample['answer_type']]
                    try:    
                        answer = get_answer_from_strategy(output_probability[i],sample,dataset[i],l=l,method=method,alpha=alpha,level=level)
                    except:
                        error_list.append(i)
                    answer_list.append((str(len(eval(sample['asr_result']))),str(len(sample['answer'])),answer,sample.answer))
                    bleu = bleu_score(sample.answer,answer)
                    referance = ' '.join(list(sample.answer))
                    rouge_answer = ' '.join(list(answer))
                    rouge_score = rouge.get_scores(rouge_answer, referance)
                    bleu_list.append(float(bleu))
                    bleu_dict[sample.answer_type].append(float(bleu))
                    rouge_dict[sample.answer_type].append(rouge_score[0])
                    #
                    # rouge_1_list.append(rouge_score[0]['rouge-1']['r'])
                    # rouge_2_list.append(rouge_score[0]['rouge-2']['r'])
                    # rouge_l_list.append(rouge_score[0]['rouge-l']['f'])
                    # rouge_result = get_rouge(answer,sample.background)
                print('error list:')
                print(error_list)
                total_rouge_list = []
                for rou in rouge_dict:
                    total_rouge_list += rouge_dict[rou]
                total_content = format_rouge_scores(merge_rouge_list(total_rouge_list))
                rouge_content_dict = {}
                for m in range(4):
                    rouge_content_dict[m] = (format_rouge_scores(merge_rouge_list(rouge_dict[m])))

                with open('predict_result/'+ 'alpha'+str(alpha)+ level+'_'+tmp+"_big_data_background_answer_l="+str(l)+str(method)+'_'+'%.3f'%(sum(bleu_list)/len(bleu_list))+'.txt', "w") as fp:
                    fp.write('*****'*5+'\n')
                    fp.write('bleu score: %f'%(sum(bleu_list)/len(bleu_list))+'\n')
                    fp.write(total_content+'\n')
                    for m in range(4):
                        fp.write('\n'+str(m)+' answer type : \n' + rouge_content_dict[m])
                        if bleu_dict[m]:
                            fp.write('bleu_score:' + str(sum(bleu_dict[m])/len(bleu_dict[m])))
                    # fp.write('rouge_1 score: %f'%(sum(rouge_1_list)/len(rouge_1_list))+'\n')
                    # fp.write('rouge_2 score: %f'%(sum(rouge_2_list)/len(rouge_2_list))+'\n')
                    # fp.write('rouge_l score: %f'%(sum(rouge_l_list)/len(rouge_l_list))+'\n')
                    fp.write(json.dumps(answer_list,ensure_ascii=False,indent=1))
                print('*****'*5)
                print(tmp+'_method_'+str(method)+':')
                print(total_content)
                print('bleu score: %f'%(sum(bleu_list)/len(bleu_list)))
                # print('rouge_1 score: %f'%(sum(rouge_1_list)/len(rouge_1_list)))
                # print('rouge_2 score: %f'%(sum(rouge_2_list)/len(rouge_2_list)))
                # print('rouge_l score: %f'%(sum(rouge_l_list)/len(rouge_l_list)))
print('using time:')
print(time.time()-start_time)



import pandas as pd
from sklearn import metrics
import numpy as np
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np
from transformers import AutoTokenizer, BertForTokenClassification, AdamW
import torch
# from transformers import DataCollatorForTokenClassification

# from datasets import load_metric
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer


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


# train_dialogue_dataset = DialogueDataset('../data/dialogue_train.tsv')
# valid_dialogue_dataset = DialogueDataset('../data/dialogue_valid.tsv')
# AutoTokenizer.from_pretrained("bert-base-chinese")
if torch.cuda.is_available():
    device = torch.device('cuda')
    local = False
else:
    device = torch.device('cpu')
    local = True
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if local:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = BertForTokenClassification.from_pretrained('bert-base-chinese')
# data_collator = DataCollatorForTokenClassification(tokenizer,max_length=512,padding='max_length')
else:
    model = BertForTokenClassification.from_pretrained('/home/hadoop-mtai/cephfs/data/mabing05/seq2seq/transformer_model')
    tokenizer = AutoTokenizer.from_pretrained('/home/hadoop-mtai/cephfs/data/mabing05/seq2seq/transformer_model/')
# train_dialogue_dataset = DialogueDataset('../data/dialogue_train.tsv')
#save model
# import os
# from transformers import WEIGHTS_NAME, CONFIG_NAME
# output_dir = '../data/transformers_model'
# model_to_save = model.module if hasattr(model, 'module') else model
# output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
# output_config_file = os.path.join(output_dir, CONFIG_NAME)
#
# torch.save(model_to_save.state_dict(), output_model_file)
# model_to_save.config.to_json_file(output_config_file)
# tokenizer.save_vocabulary(output_dir)
# train_dialogue_dataset = DialogueDataset('../data/dialogue_train.tsv')
# toy_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.valid.0.tsv')
train_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.train.0.tsv')
valid_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.valid.0.tsv')
test_dialogue_dataset = DialogueDataset('../data/split_new_processed_wokorder_remarks.test.0.tsv')


# test_dialogue_dataset = DialogueDataset('../data/dialogue_test.tsv')
# train_data_loader = DataLoader(train_dialogue_dataset,batch_size=4,shuffle=True)
# valid_data_loader = DataLoader(valid_dialogue_dataset,batch_size=4,shuffle=True)
# test_data_loader = DataLoader(test_dialogue_dataset,batch_size=4,shuffle=True)

# metric = load_metric("seqeval")

# optim = AdamW(model.parameters(), lr=5e-5)
if local:
    batch_size = 4
else:
    batch_size = 24

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
    f"bigdata_test-{'tokenclassification'}",
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

trainer = Trainer(
    model.to(device),
    args,
    train_dataset=train_dialogue_dataset,
    eval_dataset=test_dialogue_dataset,
    # data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# trainer.evaluate()
trainer.train()
# trainer.evaluate()

# trainer.save_model("model")
# print('load model in localfile')
# model = BertForTokenClassification.from_pretrained('model')

# trainer = Trainer(
#     model.to(device),
#     args,
#     train_dataset=toy_dialogue_dataset,
#     eval_dataset=toy_dialogue_dataset,
#     # data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )
# #get strategy from training dataset
# dataset = toy_dialogue_dataset
# result = trainer.predict(dataset)

# get answer from output probability of token classification layer
# def get_answer_from_strategy(output_probability,sample,tokenized_result,method=0):
#     #0: fixed length answer 1: flexible length answer 2:parameter length answer
#
#     if method == 0: #fixed length answer
#         l = 5
#         input_length = int(tokenized_result.attention_mask.sum())
#         density_list = [output_probability[i:i+l].sum() for i in range(input_length-l)]
#         answer_start = np.argmax(density_list)
#         answer_list = tokenized_result.tokens()[answer_start:answer_start+l]
#         while '&' in answer_list or '[CLS]' in answer_list or '[SEP]' in answer_list or '[PAD]' in answer_list:
#             density_list[answer_start] = 0
#             answer_start = np.argmax(density_list)
#             answer_list = tokenized_result.tokens()[answer_start:answer_start + l]
#         return ''.join(answer_list)
# output_probability = torch.tensor(result.predictions).softmax(dim=-1)[:,:,1]
# for i in range(len(dataset)):
#     sample = dataset.dataset.loc[i]
#     answer = get_answer_from_strategy(output_probability[i],sample,dataset[i])
#     blue = get_blue(answer,sample.background)
#     rouge_result = get_rouge(answer,sample.background)








# print('over')
# result = trainer.predict(toy_dialogue_dataset)
# label = result.label_ids
#
# trainer.evaluate()
# for epoch in range(3):
#     for batch in valid_data_loader:
#         optim.zero_grad()
#         inputs = tokenizer(batch['question_summary'],batch['dialogue'],add_special_tokens=True,padding='max_length',max_length=512,return_tensors='pt')
#         outputs = model(**inputs,labels=batch['summary_label'])
#         loss = outputs[0]
#         loss.backward()
#         optim.step()




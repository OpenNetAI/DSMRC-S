import pandas as pd
import re

class Process():
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep='\t')
        self.file_name = path_to_file.split('/')[-1]
        print('原dataset共%d行'%len(self.dataset))
        self._build_new_df()
        # self.dataset = self.dataset[[row['content']==eval(row['extend_fields_map']).get('236') for i,row in self.dataset.iterrows()]]
        # print('删除extend_fields_map字段不匹配之后剩%d行'%len(self.dataset))
        # self.train_dataset = pd.read_csv('../data/big_data/mabing_workorder_remarks.train.0.tsv',sep='\t')
        # tmp = 0
        # tmp_1 = 0
        # for i,row in self.train_dataset.iterrows():
        #     if row['content'] != eval(row['extend_fields_map']).get('236'):   #75.8w含18.46w
        #         tmp += 1
        #     if not eval(row['extend_fields_map']).get('236'):    #16.7w
        #         tmp_1 += 1
        # print(tmp)
    def _build_new_df(self):
        #answer_type = {0:background,1:purpose,2:key,3:solved_method}
        #question_dict = {0:'对话的背景是什么',1:'来电的诉求是什么',2:'问题的关键是什么',3:'解决方案是什么'}
        res = pd.DataFrame(columns=('asr_result','answer_type','answer','token_label'))
        for i, row in self.dataset.iterrows():
            background,purpose,key,solved_method = '','','',''
            asr_result = eval(row['asr_result'])
            dialogue_l = [j['text'] for j in asr_result]
            dialogue = '&'.join(dialogue_l)
            fields_map = eval(row['extend_fields_map'])
            content_parse = re.findall('.*背景：(.*)（必填）.*诉求：(.*)', row['content'])
            if content_parse and len(content_parse[0]) == 2:
                background, purpose = content_parse[0]
            else:
                if fields_map.get('236'):
                    background_and_purpose = fields_map.get('236')
                    content_parse = re.findall('.*背景：(.*)（必填）.*诉求：(.*)',background_and_purpose)
                    if content_parse and len(content_parse[0]) == 2:
                        background,purpose = content_parse[0]
                    else:
                        print('background error')
                        print(row)
            if fields_map.get('5'):
                key_and_solved_method = fields_map.get('5')
                content_parse = re.findall('.*关键处理过程.*：(.*?)（必填.*解决方案：(.*?)（必填）.*',key_and_solved_method)
                if content_parse and len(content_parse[0]) == 2:
                    key,solved_method = content_parse[0]
                else:
                    content_parse = re.findall('.*关键处理过程.*）(.*?)（必填.*解决方案：(.*?)（必填）.*', key_and_solved_method)
                    if content_parse and len(content_parse[0]) == 2:
                        key, solved_method = content_parse[0]
                    else:
                        content_parse = re.findall('.*关键处理过程.*：(.*?)解决方案：(.*?)用户认可解决方案客户是否认可.*', key_and_solved_method)
                        if content_parse and len(content_parse[0]) == 2:
                            key, solved_method = content_parse[0]
                        else:
                            content_parse = re.findall('.*关键处理过程.*：(.*?)必填）.*解决方案：(.*?)(必填).*解决方案客户是否认可.*', key_and_solved_method)
                            if content_parse and len(content_parse[0]) == 2:
                                key, solved_method = content_parse[0]
                            else:
                                content_parse = re.findall('.*关键处理过程.*：(.*?)解决方案：(.*?)解决方案客户是否认可.*', key_and_solved_method)
                                if content_parse and len(content_parse[0]) == 2:
                                    key, solved_method = content_parse[0]
                                else:
                                    content_parse = re.findall('.*关键处理过程.*：(.*?)（必填）.*解决方案(.*?)（必填）.*解决方案客户是否认可.*',
                                               key_and_solved_method)
                                    if content_parse and len(content_parse[0]) == 2:
                                        key, solved_method = content_parse[0]
                                    else:
                                        print('key error')
                                        print(row)
            else:
                content = row['content']
                content_parse = re.findall('.*关键处理过程.*：(.*?)（必填.*解决方案：(.*?)（必填）.*',content)
                if content_parse and len(content_parse[0]) == 2:
                    key,solved_method = content_parse[0]
            background_token_label = [[0]*len(j) for j in dialogue_l]
            purpose_token_label = [[0]*len(j) for j in dialogue_l]
            key_token_label = [[0]*len(j) for j in dialogue_l]
            solved_method_token_label = [[0]*len(j) for j in dialogue_l]

            for j,dia in enumerate(dialogue_l):
                for k,char in enumerate(dia):
                    if char in background:
                        background_token_label[j][k] = 1
                    if char in purpose:
                        purpose_token_label[j][k] = 1
                    if char in key:
                        key_token_label[j][k] = 1
                    if char in solved_method:
                        solved_method_token_label[j][k] = 1
            while len(dialogue) + 3 + 15 > 512:
                if not self._end_is_spare(background_token_label) and not self._end_is_spare(purpose_token_label) and \
                    not self._end_is_spare(key_token_label) and not self._end_is_spare(solved_method_token_label):
                    break
                if not background and not purpose and not key and not solved_method:
                    break
                if background and self._end_is_spare(background_token_label):
                    background_token_label.pop()
                else:
                    background = ''
                if purpose and self._end_is_spare(purpose_token_label):
                    purpose_token_label.pop()
                else:
                    purpose = ''
                if key and self._end_is_spare(key_token_label):
                    key_token_label.pop()
                else:
                    key = ''
                if solved_method and self._end_is_spare(solved_method_token_label):
                    solved_method_token_label.pop()
                else:
                    solved_method = ''
                dialogue_l.pop()
                asr_result.pop()
            if not background and not purpose and not key and not solved_method:
                continue

            if not background or len([j for j in background if j in dialogue])/len(background)<0.5:
                background = ''
            if not purpose or len([j for j in purpose if j in dialogue])/len(purpose)<0.5:
                purpose = ''
            if not key or len([j for j in key if j in dialogue])/len(key)<0.5:
                key = ''
            if not solved_method or len([j for j in solved_method if j in dialogue])/len(solved_method)<0.5:
                solved_method = ''
            if background:
                token_label = []
                for tmp in background_token_label:
                    token_label += tmp
                    token_label += [0]
                token_label.pop()
                assert len('&'.join([j['text'] for j in asr_result])) == len(token_label)
                res = res.append({'asr_result':asr_result,'answer_type':0,'answer':background,'token_label':token_label},ignore_index=True)
            if purpose:
                token_label = []
                for tmp in purpose_token_label:
                    token_label += tmp
                    token_label += [0]
                token_label.pop()
                assert len('&'.join([j['text'] for j in asr_result])) == len(token_label)
                res = res.append({'asr_result':asr_result,'answer_type':1,'answer':purpose,'token_label':token_label},ignore_index=True)
            if key:
                token_label = []
                for tmp in key_token_label:
                    token_label += tmp
                    token_label += [0]
                token_label.pop()
                assert len('&'.join([j['text'] for j in asr_result])) == len(token_label)
                res = res.append({'asr_result':asr_result,'answer_type':2,'answer':key,'token_label':token_label},ignore_index=True)
            if solved_method:
                token_label = []
                for tmp in solved_method_token_label:
                    token_label += tmp
                    token_label += [0]
                token_label.pop()
                assert len('&'.join([j['text'] for j in asr_result])) == len(token_label)
                res = res.append({'asr_result':asr_result,'answer_type':3,'answer':solved_method,'token_label':token_label},ignore_index=True)
        res.to_csv('../data/processed_'+self.file_name,sep='\t')

    def _end_is_spare(self, something_list):
        if len(something_list) <= 2:
            return False
        if sum(something_list[-1]) <= 1:
            return True
        if sum(something_list[-1]) / sum([sum(i) for i in something_list]) <= 0.2:
            return True
        return False



    def _refine_dataset(self):
        asr_result = self.dataset['asr_result']
        dialogue_list = []
        # speaker_list =
        for each_asr in asr_result:
            eval_asr = eval(each_asr)
            dialogue = '&'.join([i['text'] for i in eval_asr])
            speaker = [i['channel'] for i in eval_asr]
            dialogue_list.append(dialogue)






Process('../data/big_data/mabing_workorder_remarks.test.0.tsv')

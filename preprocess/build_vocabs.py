import argparse
import numpy as np
import pandas as pd
from konlpy.tag import Mecab
import os
import json

def build_vocab(args):
    if args.mode in ["train", "val"]:
        json_data=pd.read_json('{}/train/라벨링데이터/생활안전/대본X/output.json'.format(args.video_dir))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/생활안전/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/스포츠/대본X/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본X/output.json'.format(args.video_dir)))

        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/생활안전/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/스포츠/대본X/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본O/output.json'.format(args.video_dir)))
        json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본X/output.json'.format(args.video_dir)))

    # data 랜덤하게 split하기 위해서 permutation 사용.
    # json_data = json_data.iloc[np.random.RandomState(seed=args.seed).permutation(len(json_data))]
    questions = list(json_data['que'])
    answer_candidates = np.asarray(json_data['answers'])
    summary = list(json_data['sum'])
    m = Mecab().morphs
    script=[]

    # script 정보 load
    init_script = list(json_data['script'])
    script_exi = list(json_data['script_exi'])
    for idx, exi in enumerate(script_exi):
        if exi == 1:
            script.append(init_script[idx])

    print(answer_candidates.shape)

    print('number of questions: %s' % len(questions))

    if args.mode in ['train']:
        print('Building vocab')

        answer_token_to_idx = {'<UNK0>':0, '<UNK1>':1 } # anwer_candidate에 대한 token 저장할 dictionary
        question_token_to_idx = {'<NULL>':1, '<UNK>':1} # questions에 대한 token 저장할 dictionary
        summ_token_to_idx = {'<NULL>':1, '<UNK>':1} # sum에 대한 token 저장할 dictionary
        question_answer_token_to_idx = {'<NULL>':0 , '<UNK>': 1} # question, answer, sum에 대한 token 저장할 dictionary
        script_token_to_idx = {'<NULL>':1, '<UNK>':1} # script에 대한 token 저장할 dictionary

        # 정답 후보에 대한 tokenize
        for candidates in answer_candidates:
            for answer in candidates:
                for token in m(answer):
                    if token not in answer_token_to_idx:
                        answer_token_to_idx[token] = len(answer_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        print('Get answer_token_to_idx, num %d' % len(answer_token_to_idx))

        # 질문에 대한 tokenize
        for question in questions:
            for token in m(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(answer_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)

        # 요약문에 대한 tokenize
        for summ in summary:
            for token in m(summ):
                if token not in summ_token_to_idx:
                    summ_token_to_idx[token] = len(summ_token_to_idx)
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)                  
        
        # 대본에 대한 tokenize
        for sc in script:
            if script is not None or script is not "NaN":
                for token in m(sc):
                    if token not in script_token_to_idx:
                        script_token_to_idx[token] = len(script_token_to_idx)
                    if token not in question_answer_token_to_idx:
                        question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
        

        print('Get answer_token_to_idx')
        print(len(answer_token_to_idx))
        print('Get summ_token_to_idx')
        print(len(summ_token_to_idx))
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))
        print('Get script_token_to_idx')
        print(len(script_token_to_idx))

        # vocab 생성
        vocab = {
        'question_token_to_idx': question_token_to_idx,
        'answer_token_to_idx': answer_token_to_idx,
        'sum_token_to_idx': summ_token_to_idx,
        'question_answer_token_to_idx' : question_answer_token_to_idx,
        'script_token_to_idx': script_token_to_idx
        }

        vocab_save_directory = args.vocab_json.format(args.dataset, args.dataset)
        print("Write into %s" % vocab_save_directory)

        with open(vocab_save_directory, 'w') as file:
            json.dump(vocab, file, indent=4)
            
        return vocab
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa','video-narr'], type=str)
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--mode', choices=['train', 'val', 'test'])
    parser.add_argument('--seed', type=int, default=153)
    parser.add_argument('--video_dir', help='base directory of data')

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'video-narr':
        # args.output_pt = 'data/{}/{}_{}_questions.pt' if args.split_train else 'data/{}/{}_{}all_questions.pt'
        # args.vocab_json = 'data/{}/{}_vocab.json'
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        build_vocab(args)
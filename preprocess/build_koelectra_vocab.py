import torch
from transformers import ElectraModel, ElectraTokenizer, AutoTokenizer
import numpy as np
import pandas as pd
from konlpy.tag import Mecab
from tqdm import tqdm
import argparse
import json
import os


def todevice(device, tokenized):
    return {k:v.to(device) for k,v in tokenized.items()}

def get_hidden_states(encoded, model, device):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    
    # with torch.no_grad():
    #     output = model(**encoded, output_hidden_states = True)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    last_hidden_state, hidden_states, attentions = model(**todevice(device, encoded),
                                                                    output_hidden_states = True, 
                                                                    output_attentions=True,
                                                                    return_dict=False) # return_dict=True 시 outputs = model(**pt)로 return 받아야

    # # 512 words
    # print('last hidden state shape:', last_hidden_state.shape)

    # # 13:(initial embeddings + 12 BERT layers)
    # print('hidden state shape:', torch.stack(hidden_states).size()) # (layer, N, T, D)

    # print('attentions shape:', torch.stack(attentions).size())

    # Remove batch-dim, Swap dimensions 0 and 1.
    token_embeddings = torch.stack(hidden_states).permute(1,2,0,3) # (N, T, layer, D)

    #https://is-rajapaksha.medium.com/bert-word-embeddings-deep-dive-32f6214f02bf

    # initial embeddings can be taken from 0th layer of hidden states
    # hidden_states = (maxlen, layers, dimension)
    word_embed_2 = token_embeddings[:,:,0,:] # [N, T, L, D] -> [N, T, D]
    # sum of all hidden states
    word_embed_3 = token_embeddings.sum(2) # [N, T, L, D] -> [N, T, D]
    # sum of second to last layer
    word_embed_4 = token_embeddings[:,:,2:,:].sum(2) # [N, T, 13, D] -> [N, T, 11, D] -> [N, T, D]
    # sum of last four layer
    word_embed_5 = token_embeddings[:,:,-4:,:].sum(2) # [N, T, 4, D] -> [N, T, D]
    # concatenate last four layers
    word_embed_6 = torch.cat([token_embeddings[:,:,i,:] for i in range(-4, -1+1)], dim=1) # [N, T, 13, D] -> [N, T, 4, D] -> [N, T, 4*D]
    return word_embed_2, word_embed_3, word_embed_4, word_embed_5, word_embed_6

    # # Get all hidden states
    # states = output.hidden_states # 13, 1, 512, 768

    # # Stack and sum all requested layers
    # output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # return output

def get_word_idx(sent: str, word: str):
    # return Mecab().morphs.index(word)
    return sent.index(word)

def get_orgword_vector(sent, word, vocab_idx, tokenizer, model, device, pool_method:int):
    # sent : tokenized and joined with space; ex. "나 는 학교 에 간 다" or "학교"
    # sent = ' '.join(mecab('우박이라고함'))
    
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")

    if vocab_idx > len(tokenizer):
        idx = get_word_idx(sent, word)
        token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    else:
         token_ids_word = (np.array([1]),) # [CLS] token

    output = get_hidden_states(encoded, model, device)
    output = output[pool_method-1]

    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word].squeeze()
    return word_tokens_output if len(token_ids_word[0])==1 else word_tokens_output.mean(dim=0)

def get_orgword_vector_batch(sents, words, tokenizer, model, device, pool_method:int):
    # sent : tokenized and joined with space; ex. "나 는 학교 에 간 다" or "학교"
    # sent = ' '.join(mecab('우박이라고함'))
    
    encoded_batch = tokenizer.batch_encode_plus(sents, return_tensors="pt",padding=True)
    output = get_hidden_states(encoded_batch, model, device)
    output = output[pool_method-1]
    
    token_ids_word_li = []
    for i in range(len(sents)):
        idx = get_word_idx(sents[i], words[i])
        token_ids_word = np.where(np.array(encoded_batch.word_ids(i)) == idx)
        token_ids_word_li.append(token_ids_word)
        
    # Only select the tokens that constitute the requested word
    word_tokens_outputs = []
    for i in range(len(token_ids_word_li)):
        word_tokens_output = output[i, token_ids_word_li[i][0],:].mean(dim=0).squeeze()
        word_tokens_outputs.append(word_tokens_output)
    word_tokens_outputs = torch.stack(word_tokens_outputs)
    return word_tokens_outputs

def get_word_vector_batch(words, tokenizer, model, device, pool_method:int):
    # sent : tokenized and joined with space; ex. "나 는 학교 에 간 다" or "학교"
    # sent = ' '.join(mecab('우박이라고함'))
    
    encoded_batch = tokenizer.batch_encode_plus(words, return_tensors="pt", padding=True)
    output = get_hidden_states(encoded_batch, model, device)


    output = output[pool_method-1]
    token_ids_word = (np.array([1]),) # [CLS] token

    # Only select the tokens that constitute the requested word
    word_tokens_output = output[:,token_ids_word,:].squeeze()
    return word_tokens_output

def build_vocab(args=None, save_vocab=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model.eval()
    model.to(device)
    # tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator", use_fast=True)
    print(
    tokenizer.tokenize("천둥번개를 동반한 소나기가 34번 내렸습니다. 상당히 많이 내렸는데"),'\n',
    tokenizer.tokenize("장갑을 낀 사람이 밭에 있다\n그 사람이 허리를 숙이고 있다\n그 사람이 작업을 한다"))

    video_dir = './data/raw_data' if not args else args.video_dir
    json_data=pd.read_json('{}/train/라벨링데이터/생활안전/대본X/output.json'.format(video_dir))
    json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/생활안전/대본O/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/스포츠/대본X/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본O/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/train/라벨링데이터/예능교양/대본X/output.json'.format(video_dir)))

    json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/생활안전/대본O/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/스포츠/대본X/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본O/output.json'.format(video_dir)))
    json_data=json_data.append(pd.read_json('{}/test/라벨링데이터/예능교양/대본X/output.json'.format(video_dir)))

    answer_candidates = [answer for candidates in json_data['answers'] for answer in candidates]
    
    # 분절되는 answer candidate 저장
    mecab = Mecab().morphs
    new_tokens = []

    for answer in answer_candidates:

        for morph in mecab(answer):
            
            answer_tokened = tokenizer.tokenize(morph)

            if len(answer_tokened)==1 or morph in new_tokens:
                continue

            # subword 분절 여부 check -> 해당 단어 저장
            #print(morph, answer_tokened)
            new_tokens.append(morph)

    # embedding matrix 생성
    print('Embedding Matrix 생성 중 (KoElectra Tokenizer) ...')

    embsize = model.config.embedding_size
    init_vocab_cnt = len(tokenizer)
    final_vocab_cnt = init_vocab_cnt +len(new_tokens)

    matrix = np.zeros((final_vocab_cnt, embsize))
    w2idx = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    w2idx += [(w, idx) for idx, w in enumerate(new_tokens, init_vocab_cnt)]

    vocab_words = [w for w, _ in w2idx]

    batch_size = 3000
    for i in tqdm(range(0, init_vocab_cnt, batch_size)):
        word_list = vocab_words[i:min(i+batch_size, init_vocab_cnt)]
        wordvecs = get_word_vector_batch(word_list, tokenizer, model, device, 4)
        if i==0:
            print(wordvecs.size())
        matrix[i:min(i+batch_size, init_vocab_cnt), :] = wordvecs.detach().cpu().numpy()

    for i in tqdm(range(init_vocab_cnt, final_vocab_cnt, batch_size)):
        word_list = vocab_words[i:min(i+batch_size, final_vocab_cnt)]
        wordvecs = get_orgword_vector_batch(word_list, word_list, tokenizer, model, device, 4)
        matrix[i:i+batch_size, :] = wordvecs.detach().cpu().numpy()

    # tokenizer에 분절되던 token 추가
    _ = tokenizer.add_tokens(new_tokens)
    final_vocab_cnt = len(tokenizer)
    embsize = model.config.embedding_size

    if save_vocab:
        vocab_save_fname = args.tokenizer_dir + f'tokenizer_vocab_{final_vocab_cnt}_{embsize}.json'
        print("Write into %s" % vocab_save_fname)
        with open(vocab_save_fname, 'w') as file:
            json.dump(tokenizer.vocab, file, indent=4)
        
    print(tokenizer.save_pretrained(args.tokenizer_dir))
    np.save(args.tokenizer_dir+'vocab_matrix.npy', matrix)
    print('KoElectra Vocab matrix saved in', args.tokenizer_dir+'vocab_matrix.npy')
        # model.resize_token_embeddings(len(tokenizer))
    print('KoElectra Vocab matrix :', matrix.shape)
    return tokenizer, matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tgif-qa', choices=['tgif-qa', 'msrvtt-qa', 'msvd-qa','video-narr'], type=str)
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab.json')
    parser.add_argument('--video_dir', help='base directory of data')
    parser.add_argument('--tokenizer_dir', help='base directory of tokenizer to save')

    args = parser.parse_args()

    if args.dataset == 'video-narr':
        # args.output_pt = 'data/{}/{}_{}_questions.pt' if args.split_train else 'data/{}/{}_{}all_questions.pt'
        # args.vocab_json = 'data/{}/{}_vocab.json'
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))

    build_vocab(args)

import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from glob import glob
from termcolor import colored

from DataLoader import VideoQADataLoader
from utils import todevice

import model.HCRN as HCRN

from config import cfg, cfg_from_file

def get_model_path(dir_name, metric = 'acc'):
    ckpt_paths = glob(dir_name+'/modelall_*.pt')
    sort_by_max = lambda fname : float(fname.split('_')[1].replace(metric,''))
    model_path = sorted(ckpt_paths, key = sort_by_max, reverse=True)[0]
    return model_path

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)

# 검증 및 추론 함수
def validate(cfg, model, data, device, write_preds=False, istrain=False):
    model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    v_ids = []
    q_ids = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            batch_size = answers.size(0)
            logits = model(*batch_input).to(device)
            if cfg.dataset.question_type in ['action', 'transition','none']:
                if istrain:
                    batch_agg = np.concatenate(np.tile(np.arange(batch_size).reshape([batch_size, 1]),
                                                       [1, 5])) * 5  # [0, 0, 0, 0, 0, 5, 5, 5, 5, 1, ...]
                    answers_agg = tile(answers, 0, 5)
                    loss = torch.max(torch.tensor(0.0).cuda(),
                                     1.0 + logits - logits[answers_agg + torch.from_numpy(batch_agg).cuda()])
                    loss = loss.sum()
                    total_loss += loss.detach()

                preds = torch.argmax(logits.view(batch_size, 5), dim=1)
                agreeings = (preds == answers)
            elif cfg.dataset.question_type == 'count':
                answers = answers.unsqueeze(-1)
                preds = (logits + 0.5).long().clamp(min=1, max=10)
                batch_mse = (preds - answers) ** 2
            else:
                preds = logits.detach().argmax(1)
                agreeings = (preds == answers)
            if write_preds:
                if cfg.dataset.question_type not in ['action', 'transition', 'count', 'none']:
                    preds = logits.argmax(1)
                if cfg.dataset.question_type in ['action', 'transition','none']:
                    answer_vocab = data.vocab['question_answer_idx_to_token']
                else:
                    answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    if cfg.dataset.question_type in ['count', 'transition', 'action','none']:
                        all_preds.append(predict.item())
                    else:
                        all_preds.append(answer_vocab[predict.item()])
                for gt in answers:
                    if cfg.dataset.question_type in ['count', 'transition', 'action','none']:
                        gts.append(gt.item())
                    else:
                        gts.append(answer_vocab[gt.item()])
                for id in video_ids:
                    v_ids.append(id.detach().cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id)

            if cfg.dataset.question_type == 'count':
                total_acc += batch_mse.float().sum().item()
                count += answers.size(0)
            else:
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)
        acc = total_acc / count
    if not write_preds and istrain:
        return acc, total_loss
    elif not write_preds and not istrain:
        return acc
    else:
        return acc, all_preds, gts, v_ids, q_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='video_narr.yml', type=str)#default='tgif_qa_action.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['tgif-qa', 'msrvtt-qa', 'msvd-qa', 'video-narr']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    # model_ckpt = cfg.test.model_fname
    # ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', model_ckpt)

    ckpt_saved = os.path.join(cfg.dataset.save_dir, cfg.exp_name, 'ckpt')
    ckpt = get_model_path(ckpt_saved, 'acc')
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = 'test_{}_appearance_feat.h5'
        cfg.dataset.motion_feat = 'test_{}_motion_feat.h5'
        cfg.dataset.vocab_json = '{}_all_vocab.json'
        cfg.dataset.test_question_pt = '{}_test_questions.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))

    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }

    test_loader = VideoQADataLoader(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.HCRNNetwork(**model_kwargs).to(device)
    new_state_dict = {}
    for k, v in loaded['state_dict'].items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    #model.load_state_dict(loaded['state_dict'])


    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device, cfg.test.write_preds)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, f"test_preds_{cfg.exp_name.split('-')[0]}.json")

        if cfg.dataset.question_type in ['action', 'transition','none']: 
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                # org_v_names = obj['video_names']
                org_v_names = obj['video_ids']
                #org_q_ids = obj['question_id'][:len(obj['question_id'])//2]
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            if isinstance(org_q_ids, list):
                print("len of org_q_ids:", len(org_q_ids))
            else:
                print("shape of org_q_ids: " + org_q_ids.shape)
                print("dtype of org_q_ids: " + org_q_ids.dtype)
            
            # for idx in range(len(org_q_ids)+1):
            q_ids = list(map(lambda x:x.cpu(), q_ids))

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_ids[idx], questions[idx], ans_candidates[idx]]

            import collections
            if cfg.test.submission:        
                instances =  [(q_id, pred) for q_id, pred in zip(np.hstack(q_ids).tolist(), preds)]
                instances = sorted(instances, key = lambda x: x[0])
                instances = collections.OrderedDict(instances)
            else:
                instances = [
                    {'video_id': video_id, 'question_id': q_id, 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                    'answer': answer,
                    'prediction': pred} for video_id, q_id, answer, pred in
                    zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
                   
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)

            sys.stdout.write('Display 20 samples...\n')

            # Display 10 samples
            for idx in range(20):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                print('Question ID: {}'.format(q_ids[idx]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 examples
            for idx in range(20):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                print('Question ID: {}'.format(q_ids[idx]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
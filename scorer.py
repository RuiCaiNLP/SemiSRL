from __future__ import print_function
from data_utils import _PAD_, _UNK_,_BOA_, _EOA_
import inter_utils
from utils import get_torch_variable_from_np, get_data
import torch
import os
import numpy as np
import sys


def log(*args, **kwargs):
    print(file=sys.stderr, *args, **kwargs)

def get_PRF(probs, gold):
    predicts = np.argmax(probs.cpu().data.numpy(), axis=1)
    gold = gold.cpu().data.numpy()
    correct = 0.0
    NonullTruth = 0.0
    NonullPredict = 0.0
    for p, g in zip(predicts, gold):
        if g == 0:
            continue
        if g > 1:
            NonullTruth += 1
        if p > 1:
            NonullPredict += 1
        if p == g and g > 1:
            correct += 1

    return correct, NonullPredict, NonullTruth


def sem_f1_score(target, predict, argument2idx, unify_pred = False, predicate_correct=0, predicate_sum=0):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0

    if unify_pred:
        predicate_correct = 0
        predicate_sum = 0


    for i in range(len(target)):
        for j in range(len(target[i])):
            pred_i = predict[i][j]
            golden_i = target[i][j]

            if unify_pred and j == 0:
                predicate_sum += 1
                if pred_i == golden_i:
                    predicate_correct += 1
            else:
                if golden_i == argument2idx[_PAD_]:
                    continue
                total += 1
                if pred_i == argument2idx[_UNK_]:
                    pred_i = argument2idx['_']
                if golden_i == argument2idx[_UNK_]:
                    golden_i = argument2idx['_']
                if pred_i != argument2idx['_']:
                    predict_args += 1
                if golden_i != argument2idx['_']:
                    golden_args += 1
                if golden_i != argument2idx['_'] and pred_i == golden_i:
                    correct_args += 1
                if pred_i == golden_i:
                    num_correct += 1

    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + predicate_sum + 1e-13)

    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    log('\teval accurate:{:.2f} predict:{} golden:{} correct:{} P:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args, correct_args, P*100, R*100, F1*100, NP*100, NR *100, NF1 * 100))

    return (P, R, F1, NP, NR, NF1)


def eval_train_batch(epoch,batch_i,loss,golden_batch,predict_batch,argument2idx):
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    batch_total = 0
    for i in range(len(golden_batch)):
        pred_i = predict_batch[i]
        golden_i = golden_batch[i]
        if golden_i == argument2idx[_PAD_]:
            continue
        batch_total += 1
        if pred_i == argument2idx[_UNK_]:
            pred_i = argument2idx['_']
        if golden_i == argument2idx[_UNK_]:
            golden_i = argument2idx['_']
        if pred_i != argument2idx['_']:
            predict_args += 1
        if golden_i != argument2idx['_']:
            golden_args += 1
        if golden_i != argument2idx['_'] and pred_i == golden_i:
            correct_args += 1
        if pred_i == golden_i:
            num_correct += 1

    log('epoch {} batch {} loss:{:4f} accurate:{:.2f} predict:{} golden:{} correct:{}'.format(epoch, batch_i, loss, num_correct/batch_total*100, predict_args, golden_args, correct_args))


def eval_data(model, elmo, dataset, batch_size ,word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2argument, idx2word, unify_pred = False, predicate_correct=0, predicate_sum=0):

    model.eval()
    golden = []
    predict = []

    output_data = []
    cur_sentence = None
    cur_sentence_data = None

    correct_pos, NonullPredict_pos, NonullTruth_pos = 0.1, 0.1, 0.
    correct_PI, NonullPredict_PI, NonullTruth_PI = 0., 0., 0.
    correct_deprel, NonullPredict_deprel, NonullTruth_deprel = 0., 0., 0.
    correct_link, NonullPredict_link, NonullTruth_link = 0., 0., 0.


    for batch_i, input_data in enumerate(inter_utils.get_batch(dataset, batch_size, word2idx,
                                                             lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2word)):
        
        target_argument = input_data['argument']
        
        flat_argument = input_data['flat_argument']

        gold_pos = input_data['gold_pos']

        gold_PI = input_data['predicates_flag']

        gold_deprel = input_data['sep_dep_rel']

        gold_link = input_data['sep_dep_link']

        target_batch_variable = get_torch_variable_from_np(flat_argument)
        gold_pos_batch_variable = get_torch_variable_from_np(gold_pos)
        gold_PI_batch_variable = get_torch_variable_from_np(gold_PI)
        gold_deprel_batch_variable = get_torch_variable_from_np(gold_deprel)
        gold_link_batch_variable = get_torch_variable_from_np(gold_link)

        sentence_id = input_data['sentence_id']
        predicate_id = input_data['predicate_id']
        word_id = input_data['word_id']
        sentence_len =  input_data['sentence_len']
        seq_len = input_data['seq_len']
        bs = input_data['batch_size']
        psl = input_data['pad_seq_len']
        
        out, out_pos, out_PI, out_deprel, out_link = model(input_data, elmo)

        a, b, c = get_PRF(out_pos, gold_pos_batch_variable.view(-1))
        correct_pos += a
        NonullPredict_pos += b
        NonullTruth_pos += c

        a, b, c = get_PRF(out_PI, gold_PI_batch_variable.view(-1))
        correct_PI += a
        NonullPredict_PI += b
        NonullTruth_PI += c

        a, b, c = get_PRF(out_deprel, gold_deprel_batch_variable.view(-1))
        correct_deprel += a
        NonullPredict_deprel += b
        NonullTruth_deprel += c

        a, b, c = get_PRF(out_link, gold_link_batch_variable.view(-1))
        correct_link += a
        NonullPredict_link += b
        NonullTruth_link += c


        _, pred = torch.max(out, 1)

        pred = get_data(pred)

        pred = np.reshape(pred, target_argument.shape)

        for idx in range(pred.shape[0]):
            predict.append(list(pred[idx]))
            golden.append(list(target_argument[idx]))

        pre_data = []
        for b in range(len(seq_len)):
            line_data = ['_' for _ in range(sentence_len[b])]
            for s in range(seq_len[b]):
                wid = word_id[b][s]
                line_data[wid-1] = idx2argument[pred[b][s]]
            pre_data.append(line_data)

        for b in range(len(sentence_id)):
            if cur_sentence != sentence_id[b]:
                if cur_sentence_data is not None:
                    output_data.append(cur_sentence_data)
                cur_sentence_data = [[sentence_id[b]]*len(pre_data[b]),pre_data[b]]
                cur_sentence = sentence_id[b]
            else:
                assert cur_sentence_data is not None
                cur_sentence_data.append(pre_data[b])

    if cur_sentence_data is not None and len(cur_sentence_data)>0:
        output_data.append(cur_sentence_data)
    
    score = sem_f1_score(golden, predict, argument2idx, unify_pred, predicate_correct, predicate_sum)
    P = correct_pos / NonullPredict_pos
    R = correct_pos / NonullTruth_pos
    F = 2 * P * R / (P + R)
    log("POS: ", P, R, F)

    P = correct_PI / NonullPredict_PI
    R = correct_PI / NonullTruth_PI
    F = 2 * P * R / (P + R)
    log("PI: ", P, R, F)

    P = correct_deprel / NonullPredict_deprel
    R = correct_deprel / NonullTruth_deprel
    F = 2 * P * R / (P + R)
    log("deprel: ", P, R, F)

    P = correct_link / NonullPredict_link
    R = correct_link / NonullTruth_link
    F = 2 * P * R / (P + R)
    log(correct_link, NonullPredict_link, NonullTruth_link)
    log("link: ", P, R, F)

    model.train()

    return score, output_data
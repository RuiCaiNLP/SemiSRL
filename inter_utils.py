from data_utils import _PAD_,_UNK_,_ROOT_,_NUM_
import math
import numpy as np
import random

def pad_batch(batch_data, batch_size, pad_int):
    if len(batch_data) < batch_size:
        batch_data += [[pad_int]] * (batch_size - len(batch_data))
    max_length = max([len(item) for item in batch_data])
    return [item + [pad_int]*(max_length-len(item)) for item in batch_data]

char_file = open('char.voc.conll2009', 'r')
char_dict = {}
idx = 0
for char in char_file.readlines():
    char_dict[char.strip()] = idx
    idx += 1

print(char_dict['c'])


def get_batch(input_data, batch_size, word2idx, lemma2idx, pos2idx, pretrain2idx, deprel2idx, argument2idx, idx2word, shuffle=False):

    if shuffle:
        random.shuffle(input_data)

    for batch_i in range(int(math.ceil(len(input_data)/batch_size))):
        
        start_i = batch_i * batch_size
        end_i = start_i + batch_size
        if end_i > len(input_data):
            end_i = len(input_data)

        data_batch = input_data[start_i:end_i]


        sentence_id_batch = [sentence[0][0] for sentence in data_batch]
        predicate_id_batch = [sentence[0][1] for sentence in data_batch]
        setence_len_batch = [int(sentence[0][2]) for sentence in data_batch]
        id_batch = [[int(item[3]) for item in sentence] for sentence in data_batch]
        index_batch = [[int(item[4]) for item in sentence] for sentence in data_batch]

        seq_len_batch = [len(sentence) for sentence in data_batch]

        flag_batch = [[int(item[5]) for item in sentence] for sentence in data_batch]
        pad_flag_batch = np.array(pad_batch(flag_batch, batch_size, 0),dtype=int)


        sentence_flags_batch = [[int(item[16])+1 for item in sentence] for sentence in data_batch]
        pad_sentence_flags_batch = np.array(pad_batch(sentence_flags_batch, batch_size, 0),dtype=int)

        predicates_idx_batch = []
        for sentence in data_batch:
            for id, item in enumerate(sentence):
                if int(item[5]) == 1:
                    predicates_idx_batch.append(id)
                    break

        text_batch = [[item[6] for item in sentence] for sentence in data_batch]
        if len(text_batch) < batch_size:
            text_batch += [[_PAD_]] * (batch_size - len(text_batch))

        word_batch = [[word2idx.get(item[6],word2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_word_batch = np.array(pad_batch(word_batch, batch_size, word2idx[_PAD_]))

        _, sen_max_len = pad_word_batch.shape
        flat_word_batch = pad_word_batch.ravel()
        char_batch = [[char_dict[c] if char_dict.has_key(c) else 0 for c in idx2word[word]] for word in flat_word_batch]
        pad_char_batch = np.array(pad_batch(char_batch, batch_size*sen_max_len, 0)).reshape(batch_size, sen_max_len, -1)

        lemma_batch = [[lemma2idx.get(item[7],lemma2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_lemma_batch = np.array(pad_batch(lemma_batch, batch_size, lemma2idx[_PAD_]))

        pos_batch = [[pos2idx.get(item[8],pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pos_batch = np.array(pad_batch(pos_batch, batch_size, pos2idx[_PAD_]))

        gold_pos_batch = [[pos2idx.get(item[13], pos2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_gold_pos_batch = np.array(pad_batch(gold_pos_batch, batch_size, pos2idx[_PAD_]))

        head_batch = [[int(item[9]) for item in sentence] for sentence in data_batch]
        pad_head_batch = np.array(pad_batch(head_batch, batch_size, -1))

        gold_head_batch = [[int(item[14]) for item in sentence] for sentence in data_batch]
        pad_gold_head_batch = np.array(pad_batch(gold_head_batch, batch_size, -1))

        rhead_batch = [[int(item[10]) for item in sentence] for sentence in data_batch]
        pad_rhead_batch = np.array(pad_batch(rhead_batch, batch_size, -1))

        deprel_batch = [[deprel2idx.get(item[11],deprel2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_deprel_batch = np.array(pad_batch(deprel_batch, batch_size, deprel2idx[_PAD_]))

        gold_deprel_batch = [[deprel2idx.get(item[15], deprel2idx[_UNK_]) for item in sentence] for sentence in
                        data_batch]
        pad_gold_deprel_batch = np.array(pad_batch(gold_deprel_batch, batch_size, deprel2idx[_PAD_]))


        sep_pad_gold_deprel_batch = pad_gold_deprel_batch
        sep_pad_gold_link_batch = pad_gold_deprel_batch
        ### constructing specific gold deprel
        for i, sentence in enumerate(data_batch):
            current_predicate_id = predicates_idx_batch[i]
            for j, item in enumerate(sentence):
                if pad_gold_head_batch[i][j]-1 == current_predicate_id:
                    sep_pad_gold_link_batch[i][j] = 3
                    continue
                if j == pad_gold_head_batch[i][current_predicate_id]-1:
                    sep_pad_gold_link_batch[i][j] = 2
                    continue
                sep_pad_gold_deprel_batch[i][j] = deprel2idx[_UNK_]
                sep_pad_gold_link_batch[i][j] = 1


        argument_batch = [[argument2idx.get(item[12],argument2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_argument_batch = np.array(pad_batch(argument_batch, batch_size, argument2idx[_PAD_]))
        flat_argument_batch = np.array([item for line in pad_argument_batch for item in line])

        pretrain_word_batch = [[pretrain2idx.get(item[6],pretrain2idx[_UNK_]) for item in sentence] for sentence in data_batch]
        pad_pretrain_word_batch = np.array(pad_batch(pretrain_word_batch, batch_size, pretrain2idx[_PAD_]))

        # flag indicies
        pad_flag_indices = [0 for _ in range(batch_size)]
        for idx in range(batch_size):
            for jdx in range(pad_flag_batch.shape[1]):
                if int(pad_flag_batch[idx, jdx]) == 1:
                    pad_flag_indices[idx] = jdx

        # children indicies
        pad_children_indicies = [[[] for _ in range(pad_rhead_batch.shape[1])] for _ in range(batch_size)]
        for idx in range(batch_size):
            for jdx in range(pad_rhead_batch.shape[1]):
                if pad_rhead_batch[idx,jdx]!=-1 and pad_rhead_batch[idx,jdx]!=0:
                    head_idx = pad_rhead_batch[idx,jdx]-1
                    pad_children_indicies[idx][head_idx].append(jdx)

        # sa relative indicies
        pad_relative_indicies = [[[[] for _ in range(2)] for _ in range(pad_rhead_batch.shape[1])] for _ in range(batch_size)]
        pad_relative_rels = [[[[] for _ in range(2)] for _ in range(pad_rhead_batch.shape[1])] for _ in range(batch_size)]
        for idx in range(batch_size):
            for jdx in range(pad_rhead_batch.shape[1]):
                if pad_rhead_batch[idx,jdx]!=-1 and pad_rhead_batch[idx,jdx]!=0:
                    head_idx = pad_rhead_batch[idx,jdx]-1
                    if head_idx < jdx:
                        pad_relative_indicies[idx][jdx][0].append(head_idx)
                        pad_relative_rels[idx][jdx][0].append(pad_deprel_batch[idx,jdx])
                    elif head_idx > jdx:
                        pad_relative_indicies[idx][jdx][1].append(head_idx)
                        pad_relative_rels[idx][jdx][1].append(pad_deprel_batch[idx,jdx])
                    for child_idx in pad_children_indicies[idx][jdx]:
                        if child_idx < jdx:
                            pad_relative_indicies[idx][jdx][0].append(child_idx)
                            pad_relative_rels[idx][jdx][0].append(pad_deprel_batch[idx,child_idx])
                        elif head_idx > jdx:
                            pad_relative_indicies[idx][jdx][1].append(child_idx)
                            pad_relative_rels[idx][jdx][1].append(pad_deprel_batch[idx,child_idx])

        pad_relative_indicies = np.array(pad_relative_indicies)
        pad_relative_rels = np.array(pad_relative_rels)

        # predicate_batch = []
        # predicate_pretrain_batch = []
        # for sentence in data_batch:
        #     predicate_idx = 0
        #     for j in range(len(sentence)):
        #         if sentence[j][3] == '1':
        #             predicate_idx = j
        #             break
        #     predicate_batch.append([word2idx.get(sentence[predicate_idx][4],word2idx[_UNK_])]*len(sentence))
        #     predicate_pretrain_batch.append([pretrain2idx.get(sentence[predicate_idx][4],pretrain2idx[_UNK_])]*len(sentence))

        # pad_predicate_batch = np.array(pad_batch(predicate_batch, batch_size, word2idx[_PAD_]))
        # pad_predicate_pretrain_batch = np.array(pad_batch(predicate_pretrain_batch, batch_size, pretrain2idx[_PAD_]))

        batch = {
            "sentence_id":sentence_id_batch,
            "predicate_id":predicate_id_batch,
            "predicates_idx":predicates_idx_batch,
            "word_id":id_batch,
            "index":index_batch,
            "flag":pad_flag_batch,
            "word":pad_word_batch,
            "char": pad_char_batch,
            "lemma":pad_lemma_batch,
            "pos":pad_pos_batch,
            "pretrain":pad_pretrain_word_batch,
            "head":pad_head_batch,
            "rhead":pad_rhead_batch,
            "deprel":pad_deprel_batch,
            "argument":pad_argument_batch,
            "flat_argument":flat_argument_batch,
            "batch_size":pad_argument_batch.shape[0],
            "pad_seq_len":pad_argument_batch.shape[1],
            "text":text_batch,
            "sentence_len":setence_len_batch,
            "seq_len":seq_len_batch,
            "origin":data_batch,
            'flag_indices':pad_flag_indices,
            'children_indicies':pad_children_indicies,
            'relative_indicies':pad_relative_indicies,
            'relative_rels':pad_relative_rels,
            'gold_pos':pad_gold_pos_batch,
            'gold_head':pad_gold_head_batch,
            'gold_deprel':pad_gold_deprel_batch,
            'predicates_flag':pad_sentence_flags_batch,
            'sep_dep_rel': sep_pad_gold_deprel_batch,
            'sep_dep_link': sep_pad_gold_link_batch,
        }

        yield batch
            


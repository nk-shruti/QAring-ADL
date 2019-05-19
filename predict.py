# from __future__ import absolute_import
import numpy as np
import argparse
import json
import os

from tqdm import tqdm
from termcolor import colored

from keras import backend as K
from keras.models import Model, load_model

from data import get_text_from_indices, get_reverse_word_index

from evaluator import f1_score, exact_match_score, metric_max_over_ground_truths

def get_best_answer_pointers(starts, ends, clip_at=None):
    if clip_at != None:
        starts = starts[:clip_at]
        ends = ends[:clip_at]

    max_ends = [ends[-1]]
    max_ends_indices = [len(ends) - 1]
    for i in range(len(ends) - 2, -1, -1):
        if ends[i] > max_ends[-1]:
            max_ends.append(ends[i])
            max_ends_indices.append(i)
        else:
            max_ends.append(max_ends[-1])
            max_ends_indices.append(max_ends_indices[-1])

    max_ends = max_ends[::-1]
    max_ends_indices = max_ends_indices[::-1]

    scores = np.asarray([s*e for s, e in zip(starts, max_ends)])

    max_score_index = np.argmax(scores)
    start, end = max_score_index, max_ends_indices[max_score_index]

    assert start <= end

    return start, end

def get_metrics(model, dev_data_gen, steps):

    print('Running model for predictions...')

    j = 0
    f1s, ems = [], []
    reverse_word_index = get_reverse_word_index()

    cc, wc = 7, 2 

    for (c, q), (a_starts, a_ends) in dev_data_gen:
        print c.shape
        print q.shape
        preds = model.predict_on_batch([c, q])
        # print q

        for j in range(len(c)):
            flag=0

            c_text = get_text_from_indices(c[j].tolist(), reverse_word_index).split()

            pred_a_start, pred_a_end = get_best_answer_pointers(preds[0][j], preds[1][j], clip_at=len(c_text))

            groundtruths = [' '.join(c_text[np.argmax(a_start[j]):np.argmax(a_end[j]) + 1]) for a_start, a_end in zip(a_starts, a_ends)]
            prediction = ' '.join(c_text[pred_a_start:pred_a_end + 1])

            f1 = metric_max_over_ground_truths(f1_score, prediction, groundtruths)
            em = metric_max_over_ground_truths(exact_match_score, prediction, groundtruths)
            f1s.append(f1)
            ems.append(em)

            if True:
                # print colored(' '.join(c_text[:pred_a_start]), 'white'), colored(' '.join(c_text[pred_a_start:pred_a_end + 1]), 'red'), colored(' '.join(c_text[pred_a_end + 1:]), 'white')
                print colored(' '.join(c_text),'white')
                                # question = get_text_from_indices(q[j].tolist(), reverse_word_index)
                # print '\n\n', colored(question, 'blue')

                # print '\n\nAnswer is : ', colored(prediction, 'yellow')

                print '\n\n'

                # for h in xrange(len(groundtruths)): print colored(groundtruths[h], 'green')
                # weights = model.get_weights()
                # single_item_model = create_model(batch_size=1)
                # single_item_model.set_weights(weights)
                while flag==0:
                    own_question = str(raw_input())
                    padding_length = len(q[j])
                    eq = encode_question(own_question,padding_length)
                    # print "----"
                    # print q[j]
                    # print "-----"
                    # print eq
                    oq = []

                    cons = []
                    cons.append(c[j])
                    cons.append(c[j])

                    


                    oq.append(eq)
                    oq.append(q[j])


                    cons = np.asarray(cons)
                    oq = np.asarray(oq)
                    # print cons.shape
                    # print oq.shape

                    pred = model.predict_on_batch([cons,oq])
                    pred_start, pred_end = get_best_answer_pointers(pred[0][0],pred[1][0],clip_at=len(c_text))
                    predicted = ' '.join(c_text[pred_start:pred_end + 1])

                    print(colored(predicted, 'yellow') )


                    print '\n\n'

                    x = raw_input("Continue with same passage? y/n")
                    if x=='n':
                        flag=1


                    if f1 > .6: cc -= 1
                    else: wc -= 1

    # print('F1 score: {}'.format(float(sum(f1s)) / len(f1s)))
    # print('EM score: {}'.format(float(sum(ems)) / len(ems)))


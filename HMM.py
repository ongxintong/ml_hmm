#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import heapq

#------part 2------
def read_train_data(dataset):
    words = []
    tags = []

    with open(dataset + "/train", "r") as f:
        lines = f.readlines()

        for line in lines:
            if line != "\n":
                pair = line.strip().split()
                words.append(pair[0])
                tags.append(pair[1])

    d = {"word": words, "tag": tags}
    df = pd.DataFrame(d)
    return df


def get_tags(traindf):
    possible_tags = traindf["tag"].tolist()
    possible_tags = list(dict.fromkeys(possible_tags))
    #     print(f"List of possible tags (of length {len(possible_tags)}) is: {possible_tags}\n")
    print(
        f"Obtained list of possible tags from the trng set - {len(possible_tags)} tags"
    )
    return possible_tags


def get_vocab(traindf):
    vocab = traindf["word"].tolist()
    vocab = list(dict.fromkeys(vocab))
    #     print(f"Training vocab (of length {len(vocab)}) is: {vocab}")
    print(f"Obtained list of vocab from the trng set - {len(vocab)} words")
    return vocab


# calc emission parameter for that specific word-tag pair
def calc_emission_pairwise(df, word, tag):
    emissiondf = (
        df.groupby(df.columns.tolist())
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    tagcount = emissiondf.groupby(["tag"]).size().reset_index(name="count")

    try:
        count_yx = emissiondf.loc[
            (emissiondf["word"] == word) & (emissiondf["tag"] == tag)
        ]["count"].values[0]
    except:
        print("no emission for the word-tag pair")
        count_yx = 0

    try:
        count_y = tagcount.loc[tagcount["tag"] == tag]["count"].values[0]
    except:
        print("tag does not exist")
        return 0

    emission_parameter = count_yx / count_y
    print(f"emission parameter is {count_yx} / {count_y} = {emission_parameter}")
    return emission_parameter


# calc emission parameter for that specific word-tag pair
# accounting for unseen words using k
def calc_emission_UNK_pairwise(df, word, tag, k=0.5):
    emissiondf = (
        df.groupby(df.columns.tolist())
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    tagcount = emissiondf.groupby(["tag"]).size().reset_index(name="count")

    if word == "#UNK#":
        count_yx = k
    else:
        try:
            count_yx = emissiondf.loc[
                (emissiondf["word"] == word) & (emissiondf["tag"] == tag)
            ]["count"].values[0]
        except:
            # print("no emission for the word-tag pair")
            count_yx = 0

    try:
        count_y = tagcount.loc[tagcount["tag"] == tag]["count"].values[0] + k
    except:
        # print("tag does not exist")
        return k

    emission_parameter = count_yx / count_y
    #     print(f'emission parameter is {count_yx} / {count_y} = {emission_parameter}')
    return emission_parameter


def read_input_data(dataset, vocabulary):
    words = []

    with open(dataset + "/dev.in", "r") as f:
        lines = f.readlines()

        for line in lines:
            word = line.strip()
            words.append(word)

    d = {"word": words}
    df = pd.DataFrame(d)
    df["unseen"] = df["word"]
    vocabulary.append("")
    # replace words outside of vocab with #UNK#
    df.loc[~df.word.isin(vocabulary), "unseen"] = "#UNK#"
    vocabulary.remove("")
    return df


# form the np array with all the calculated emission parameters for all the words and tags in the training data
def calc_all_emission_paras(dataset, traindf, possible_tags, vocab):
    emission_parameters = np.zeros(
        (len(possible_tags), len(vocab))
    )  # emission_parameters[tag][word]
    for word_idx, word in enumerate(vocab):
        if word_idx % 20 == 0:
            print(f"{word_idx}/{len(vocab)} done")
        for tag_idx, tag in enumerate(possible_tags):
            emission_parameters[tag_idx][word_idx] = calc_emission_UNK_pairwise(
                traindf, word, tag
            )

    return emission_parameters


def get_predictions_dict(possible_tags, vocab, emission_parameters):
    predict_dict = {}
    predlist = np.argmax(emission_parameters, axis=0)
    for word_idx, word in enumerate(vocab):
        tag = possible_tags[predlist[word_idx]]
        predict_dict[word] = tag

    return predict_dict

#------part 3------

def read_train_file(dataset):
    fin = open(dataset + "/train", "r", encoding="utf-8")
    data = fin.readlines()
    datalist = []
    for line in data:
        line = line.strip()
        if line == "":
            datalist.append(line)
        else:
            line = line.split(" ")
            datalist.append(line[1])
    return datalist


def transition_para(datalist, tags):
    tagcount = {}
    countu = {}
    countuv = {}
    header1 = []
    header2 = []
    tags.insert(0, "##START##")
    tags.append("##STOP##")
    for tag in tags:  # prepare empty dictionaries
        tagcount[tag] = 0
        countu[tag] = []
        countuv[tag] = [0] * (len(tags) - 1)
        header1.append(tag)
        header2.append(tag)

    header1.remove("##START##")
    header2.remove("##STOP##")
    count = 0  # count for empty spaces
    for data in datalist:
        if data == "":
            count += 1
        else:
            tagcount[data] += 1
    tagcount["##START##"] = count
    del tagcount["##STOP##"]

    for i in tags:
        countu[i] = list(tagcount.values())

    countudf = pd.DataFrame(countu, columns=header1, index=header2)

    # Get count(u,v)
    previous = ""
    current = "##START##"
    for data in datalist:
        if current != "##STOP##":
            previous = current
        else:
            previous = "##START##"

        if data != "":
            current = data
        else:
            if previous == "##START##":  # 2 empty lines = end of data
                break
            current = "##STOP##"
        idx = tags.index(previous)
        countuv[current][idx] += 1

    del countuv["##START##"]
    countuvdf = pd.DataFrame(countuv, columns=header1, index=header2)

    trans_para = countuvdf / countudf
    return trans_para

#------part 4------

def kViterbiParallel(pi, a, b, obs, topK):
    if topK == 1:
        return viterbi2(pi, a, b, obs)

    pi = pi[:-1]  # remove the extra?
    # STEP 1: Initialisation
    nStates = np.shape(b)[0]  # tags - e.g. B-NP, I-Positive
    T = np.shape(obs)[0]  # observation -> words

    #assert topK <= np.power(nStates, T), "k < nStates ^ topK"

    # delta --> highest probability of any path that reaches point i
    delta = np.zeros((T, nStates, topK))

    # phi --> argmax - aka lowest cost to reach state
    phi = np.zeros((T, nStates, topK), int)

    # The ranking of multiple paths through a state
    rank = np.zeros((T, nStates, topK), int)

    for i in range(nStates):
        delta[0, i, 0] = pi[i] * b[i, obs[0]]  # inital probability
        for k in range(0, topK):
            phi[0, i, k] = i  # set index for states

    # STEP 2: Recursion and termination
    # Go forward calculating top k scoring paths
    # for each state s1 from previous state s0 at time step t
    for t in range(1, T):  # for each word
        for s1 in range(nStates):  # for each tag
            h = []
            for s0 in range(nStates):
                for k in range(topK):
                    prob = delta[t - 1, s0, k] * a[s0, s1] * b[s1, obs[t]]
                    state = s0
                    heapq.heappush(h, (prob, state))  # retain previous state

            h_sorted = [heapq.heappop(h) for i in range(len(h))]  # heapsort on prob
            h_sorted.reverse()

            rankDict = dict()  # keep a ranking if a path crosses a state more than once

            # Retain the top k scoring paths and their phi and rankings
            for k in range(0, topK):
                delta[t, s1, k] = h_sorted[k][0]
                phi[t, s1, k] = h_sorted[k][1]

                state = h_sorted[k][1]

                if state in rankDict:
                    rankDict[state] = rankDict[state] + 1
                else:
                    rankDict[state] = 0

                rank[t, s1, k] = rankDict[state]

    h = []  # Put all the last items on the stack

    # Get the highest end prob from all the states
    for s1 in range(nStates):
        for k in range(topK):
            prob = delta[T - 1, s1, k]

            heapq.heappush(h, (prob, s1, k))  # retain previous state and k

    h_sorted = [heapq.heappop(h) for i in range(len(h))]  # heapsort on prob
    h_sorted.reverse()

    # initialize output
    path = np.zeros((topK, T), int)
    path_probs = np.zeros((topK, T), float)

    # STEP 3: Backtracking
    for k in range(topK):
        # The maximum probability and the state it came from
        max_prob = h_sorted[k][0]
        state = h_sorted[k][1]
        rankK = h_sorted[k][2]

        # Assign to output arrays
        path_probs[k][-1] = max_prob
        path[k][-1] = state

        # Then from t down to 0 store the correct sequence for t+1
        for t in range(T - 2, -1, -1):
            nextState = path[k][t + 1]  # The next state and its rank
            p = phi[t + 1][nextState][rankK]  # Get the new state
            path[k][t] = p  # Pop into output array

            # Get the correct ranking for the next phi
            rankK = rank[t + 1][nextState][rankK]

    return path, path_probs, delta, phi



def viterbi2(pi, a, b, obs):
    pi = pi[:-1] # remove the extra?
    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = np.zeros(T, int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T), float)
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T), int)

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t - 1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t - 1] * a[:, s])

    # find optimal path
    path[T - 1] = np.argmax(delta[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t + 1], [t + 1]]

    max_prob = np.max(delta[:, T - 1])

    return path, delta, phi, max_prob

#------extra functions-----
import pickle

def save_pickle(obj, name):
    with open("obj/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(name):
    with open("obj/" + name + ".pkl", "rb") as f:
        return pickle.load(f)

#----------run---------


dataset = "EN"

traindf = read_train_data(dataset)
possible_tags = get_tags(traindf)
vocab = get_vocab(traindf)

full_tags_list = read_train_file(dataset) # get full list of tags

trans_para = transition_para(full_tags_list, possible_tags)
trans_arr = trans_para.to_numpy()
a = np.delete(trans_arr, 0, 0)
a = np.delete(a,len(a[0])-1,1)
print(trans_para)
print(a)

save_pickle(trans_para, 'EN_transmissionparas')
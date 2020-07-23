import random
def readfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines()]
    tuple = [(line[0], line[1:]) for line in lines]
    random.shuffle(tuple)
    return tuple
def add_dict(lines):
    word_dict = {'<PAD>':0, '<UNK>':1}
    label_dict = dict()
    for line in lines:
        if line[0] not in label_dict:
            label_dict[line[0]] = len(label_dict)
        for word in line[1]:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict, label_dict
def padding(lines, max_len):

    return [(line[0], line[1]+(['<PAD>']*(max_len-len(line[1]))) if len(line[1])<max_len else line[1][:max_len]) for line in lines]
def to_index(lines, word_dict, label_dict):
    return [(label_dict[line[0]],
             [word_dict[word] if word in word_dict else 1 for word in line[1]]) for line in lines]
def load_data(trainfile, testfile, batch_size, max_len):
    trainlines = readfile(trainfile)
    testlines = readfile(testfile)
    word_dict, label_dict = add_dict(trainlines)
    #max_len = max([len(line[1]) for line in trainlines])
    trainlines = to_index(padding(trainlines, max_len), word_dict, label_dict)
    testlines = to_index(padding(testlines, max_len), word_dict, label_dict)
    traintuple = list(zip(*trainlines))
    testtuple = list(zip(*testlines))
    i = 0
    trainpairs = []
    while i+batch_size<=len(trainlines):
        trainpairs.append((list(traintuple[0][i:i+batch_size]), list(traintuple[1][i:i+batch_size])))
        i += batch_size
    #训练数据分批处理， 测试数据不分批
    return max_len, len(word_dict), len(label_dict), trainpairs, testtuple


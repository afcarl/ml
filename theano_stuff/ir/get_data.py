""" Get data for testing the language model """

import re

path = '/Users/benjaminbolte/Documents/ml/theano_stuff/ir/LiveQA2015-qrels-ver2.txt'

with open(path, 'r') as f:
    lines = f.readlines()
print('%d graded answers' % len(lines))

questions = dict()
answers = dict()

qpattern = re.compile('(\d+)q\t([\w\d]+)\t\t([^\t]+)\t(.*?)\t([^\t]+)\t([^\t]+)\n')
for line in lines:
    qm = qpattern.match(line)
    if qm:
        trecid = qm.group(1)
        qid = qm.group(2)
        title = qm.group(3)
        content = qm.group(4)
        maincat = qm.group(5)
        subcat = qm.group(6)
        questions[trecid] = { 'qid': qid, 'title': title, 'content': content, 'maincat': maincat, 'subcat': subcat }
        answers[trecid] = list()
    else:
        trecid, qid, score, answer, resource = line.split('`\t`')
        trecid = trecid[:-1]
        answers[trecid].append({ 'score': score, 'answer': answer, 'resource': resource })

assert len(questions) == len(answers) == 1087, 'There was an error processing the file somewhere (should have 1087 questions)'

print(questions['1129'])

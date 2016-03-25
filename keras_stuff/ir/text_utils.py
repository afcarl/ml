import os

import gensim
import itertools
from gensim.corpora import Dictionary

import xml.etree.cElementTree as ET

from keras.preprocessing.sequence import pad_sequences

import numpy as np

UNKNOWN_TOKEN = 'UNKNOWN_TOKEN'


class YahooDictionary:

    def __init__(self, source_file, vocab_size=20000, max_ans_len=1000, max_sub_len=100, max_cont_len=500, dict_file_name=''):
        assert os.path.exists(source_file), 'The file "%s" was not found' % source_file

        self.source_file = source_file
        self.vocab = Dictionary()
        self.vocab_size = vocab_size

        print('Creating XML tree...')
        tree = ET.parse(source_file)
        self.root = tree.getroot()

        # maximum lengths for everything
        self.max_ans_len = max_ans_len
        self.max_sub_len = max_sub_len
        self.max_cont_len = max_cont_len

        print('Creating dictionary...')
        self._create_dictionary()

    @staticmethod
    def tokenize(text):
        return gensim.utils.tokenize(text, to_lower=True)

    def _create_dictionary(self):
        categories = set()

        # create dictionary
        for vespaadd in self.root.iter('vespaadd'):
            doc = vespaadd.find('document')

            subject_text = YahooDictionary.tokenize(doc.find('subject').text)
            content_text = YahooDictionary.tokenize(doc.find('content').text)

            self.vocab.add_documents([subject_text, content_text], prune_at=self.vocab_size)

            # category
            categories.add(doc.find('cat').text)

            # answers
            answers = [YahooDictionary.tokenize(answer.text) for answer in doc.find('nbestanswers').getchildren()]
            self.vocab.add_documents(answers, prune_at=self.vocab_size)

        self.cat_to_idx = dict((c, i+1) for i, c in enumerate(categories))
        self.idx_to_cat = dict((i+1, c) for i, c in enumerate(categories))

    def get_docs(self):

        all_answers = []
        all_subjects = []
        all_contents = []
        all_categories = []

        # create dictionary
        for vespaadd in self.root.iter('vespaadd'):
            doc = vespaadd.find('document')

            # subject and content
            subject_text_iter = YahooDictionary.tokenize(doc.find('subject').text)
            content_text_iter = YahooDictionary.tokenize(doc.find('content').text)

            subject_enc = [self.vocab.token2id[x] for x in itertools.islice(subject_text_iter, self.max_sub_len)]
            content_enc = [self.vocab.token2id[x] for x in itertools.islice(content_text_iter, self.max_cont_len)]

            # category index
            category = self.cat_to_idx[doc.find('cat').text]

            # answers
            answers = [YahooDictionary.tokenize(answer.text) for answer in doc.find('nbestanswers').getchildren()]

            for answer in answers:
                answer_enc = [self.vocab.token2id[x] for x in itertools.islice(answer, self.max_ans_len)]

                all_categories.append(category)
                all_subjects.append(subject_enc)
                all_contents.append(content_enc)
                all_answers.append(answer_enc)

        return pad_sequences(all_answers, self.max_ans_len),\
               pad_sequences(all_subjects, self.max_sub_len),\
               pad_sequences(all_contents, self.max_cont_len),\
               np.array(all_categories)

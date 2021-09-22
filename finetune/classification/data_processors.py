import os
import csv
import json
from transformers import InputExample
from fastNLP.io.loader import LCQMCLoader, ChnSentiCorpLoader, THUCNewsLoader


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # text_a = line['keywords']
            # if len(text_a) > 0:
            #     text_a = '关键词：' + text_a
            # text_b = line['sentence']
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "100"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(119):
            labels.append(str(i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AfqmcProcessor(DataProcessor):
    """Processor for the AFQMC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = str(line['label']) if set_type != 'test' else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CmnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = str(line["label"]) if set_type != 'test' else 'neutral'
            if label == '-':
                # some example have no label, skip
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class OcnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.50k.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"] + '。'
            text_b = line["sentence2"] + '。'
            label = str(line["label"]) if set_type != 'test' else 'neutral'
            if label == '-':
                # some example have no label, skip
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CslProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = "，".join(line['keyword'])
            text_b = line['abst']
            label = str(line['label']) if set_type != 'test' else '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def _make_one_example(self, guid, text, query, query_idx, pronoun, pronoun_idx, label):
        input_text = text[:pronoun_idx] + query + text[pronoun_idx + len(pronoun):]
        return InputExample(guid=guid, text_a=input_text, text_b=None, label=label)

    def _make_one_example_v0(self, guid, text, query, query_idx, pronoun, pronoun_idx, label):
        text_a_list = list(text)
        if pronoun_idx > query_idx:
            text_a_list.insert(query_idx, "_")
            text_a_list.insert(query_idx + len(query) + 1, "_")
            text_a_list.insert(pronoun_idx + 2, "[")
            text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
        else:
            text_a_list.insert(pronoun_idx, "[")
            text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
            text_a_list.insert(query_idx + 2, "_")
            text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
        text_a = "".join(text_a_list)
        return InputExample(guid=guid, text_a=text_a, text_b=None, label=label)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            text_a_list = list(text_a)
            target = line['target']
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']
            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
            label = str(line['label']) if set_type != 'test' else 'true'
            if False and label == 'true' and set_type == 'train':
                # do data augmentation
                pronoun_list = ['她', '他', '它', '它们', '他们', '她们']
                for p in pronoun_list:
                    start = 0
                    while True:
                        pos = text_a.find(p, start)
                        if pos == -1:
                            break
                        if pos == pronoun_idx:
                            start = pos + 1
                            continue
                        examples.append(
                            self._make_one_example_v0('fake', text_a, query, query_idx, p, pos, 'false'))
                        start = pos + 1

            examples.append(
                self._make_one_example_v0(guid, text_a, query, query_idx, pronoun, pronoun_idx, label))

        # remove duplicate examples
        texts = {}
        for example in examples:
            if example.text_a in texts:
                old_example = texts[example.text_a]
                if old_example.label != example.label:
                    if old_example.guid == 'fake':
                        texts[example.text_a] = example
                    print("input: {}, label not match: {}:{}, {}:{}".format(example.text_a, old_example.guid,
                                                                            old_example.label, example.guid,
                                                                            example.label))
            else:
                texts[example.text_a] = example
        new_examples = list(texts.values())
        # print('{} origin data size: {}, new data size: {}'.format(set_type, len(lines), len(new_examples)))
        return new_examples

class ChnSentiCorpProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.data_bundle = None
        self.loader = ChnSentiCorpLoader()

    def get_train_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load()
        return self._create_examples(self.data_bundle.get_dataset('train'), 'train')

    def get_dev_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load()
        return self._create_examples(self.data_bundle.get_dataset('dev'), 'dev')

    def get_test_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load()
        return self._create_examples(self.data_bundle.get_dataset('test'), 'test')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, dataset, set_type):
        examples = []
        for (i, ins) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            text_a = ins['raw_chars']
            label = ins['target']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ThucnewsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.data_bundle = None
        self.loader = THUCNewsLoader()

    def get_train_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('train'), 'train')

    def get_dev_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('dev'), 'dev')

    def get_test_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('test'), 'test')

    def get_labels(self):
        return ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    def _create_examples(self, dataset, set_type):
        examples = []
        for (i, ins) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            text_a = ins['raw_chars']
            label = ins['target']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class LcqmcProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.data_bundle = None
        self.loader = LCQMCLoader()

    def get_train_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('train'), 'train')

    def get_dev_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('dev'), 'dev')

    def get_test_examples(self, data_dir):
        if self.data_bundle is None:
            self.data_bundle = self.loader.load(data_dir)
        return self._create_examples(self.data_bundle.get_dataset('test'), 'test')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, dataset, set_type):
        examples = []
        for (i, ins) in enumerate(dataset):
            guid = "%s-%s" % (set_type, i)
            text_a = ins['raw_chars1']
            text_b = ins['raw_chars2']
            label = ins['target']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class BQCorpusProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.data_bundle = None

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines[1:]):  # skip csv header
            guid = "%s-%s" % (set_type, i)
            text_a = ','.join(line[:-2])
            text_b = ','.join(line[-2:-1])
            # text_b = None
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            i = 2 * i
            guid1 = "%s-%s" % (set_type, i)
            guid2 = "%s-%s" % (set_type, i + 1)
            premise = line['premise']
            choice0 = line['choice0']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            choice1 = line['choice1']
            label2 = str(0 if line['label'] == 0 else 1) if set_type != 'test' else '0'
            if line['question'] == 'effect':
                text_a = premise
                text_b = choice0
                text_a2 = premise
                text_b2 = choice1
            elif line['question'] == 'cause':
                text_a = choice0
                text_b = premise
                text_a2 = choice1
                text_b2 = premise
            else:
                raise ValueError(f'unknowed {line["question"]} type')
            examples.append(
                InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
            examples.append(
                InputExample(guid=guid2, text_a=text_a2, text_b=text_b2, label=label2))
        return examples

    def _create_examples_version2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if line['question'] == 'cause':
                text_a = line['premise'] + '这是什么原因造成的？' + line['choice0']
                text_b = line['premise'] + '这是什么原因造成的？' + line['choice1']
            else:
                text_a = line['premise'] + '这造成了什么影响？' + line['choice0']
                text_b = line['premise'] + '这造成了什么影响？' + line['choice1']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

clue_processors = {
    'tnews': TnewsProcessor,
    'iflytek': IflytekProcessor,
    'cmnli': CmnliProcessor,
    'afqmc': AfqmcProcessor,
    'csl': CslProcessor,
    'wsc': WscProcessor,
    'copa': CopaProcessor,
    'ocnli': OcnliProcessor,
    'csc': ChnSentiCorpProcessor,
    'thucnews': ThucnewsProcessor,
    'bqcorpus': BQCorpusProcessor,
    'lcqmc': LcqmcProcessor
}

clue_output_modes = {
    'tnews': "classification",
    'iflytek': "classification",
    'cmnli': "classification",
    'afqmc': "classification",
    'csl': "classification",
    'wsc': "classification",
    'copa': "classification",
    'ocnli': "classification",
    'csc': "classification",
    'thucnews': "classification",
    'bqcorpus': "classification",
    'lcqmc': "classification",
}

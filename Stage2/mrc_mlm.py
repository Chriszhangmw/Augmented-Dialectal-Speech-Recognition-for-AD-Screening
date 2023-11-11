
import json, os, re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator,AutoRegressiveDecoder
# from bert4keras.snippets import open
from keras.layers import Lambda
from keras.models import Model
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

max_qp_len=64
max_a_len = 32
batch_size = 16
epochs = 10

# bert配置
config_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/zmw/big_space/zhangmeiwei_space/pre_models/tensorflow/chinese_L-12_H-768_A-12/vocab.txt'

# 标注数据
with open('./data.csv','r',encoding='utf-8') as f1:
    data1 = f1.readlines()
with open('./datageneration.csv','r',encoding='utf-8') as f2:
    data2 = f2.readlines()
    f2.close()
data1 = data1[:50]
data2 =  data2[:50]
# 保存一个随机序（供划分valid用）
if not os.path.exists('./random_order_mlm.json'):
    random_order = list(range(len(data2)))
    np.random.shuffle(random_order)
    json.dump(random_order, open('./random_order_mlm.json', 'w'), indent=4)
else:
    random_order = json.load(open('./random_order_mlm.json'))

# 划分valid
train_data = [data2[j] for i, j in enumerate(random_order) if i % 3 != 0]
valid_data = [data2[j] for i, j in enumerate(random_order) if i % 3 == 0]
train_data.extend(data1)
# train_data.extend(data1)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        """单条样本格式为
        输入：[CLS][MASK][MASK][SEP]问题[SEP]篇章[SEP]
        输出：答案
        """
        batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []
        for is_end, line in self.sample(random):
            line = line.strip().split(',')
            question = line[0]
            final_answer = line[1]
            passage = line[2]
            a_token_ids, a_segment_ids= tokenizer.encode(final_answer, maxlen=max_a_len + 1)
            qp_token_ids, qp_segment_ids= tokenizer.encode(question, passage,maxlen=max_qp_len + 1)
            token_ids = [tokenizer._token_start_id]
            token_ids += ([tokenizer._token_mask_id] * max_a_len)
            token_ids += [tokenizer._token_end_id]
            token_ids += (qp_token_ids[1:])
            segment_ids = [1] * (max_a_len+2) + qp_segment_ids[1:]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_a_token_ids.append(a_token_ids[1:])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_a_token_ids = sequence_padding(batch_a_token_ids, max_a_len)
                yield [batch_token_ids, batch_segment_ids], batch_a_token_ids
                batch_token_ids, batch_segment_ids, batch_a_token_ids = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path,
    with_mlm=True,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)
output = Lambda(lambda x: x[:, 1:max_a_len + 1])(model.output)
model = Model(model.input, output)
model.summary()


def masked_cross_entropy(y_true, y_pred):
    """交叉熵作为loss，并mask掉padding部分的预测
    """
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy


model.compile(loss=masked_cross_entropy, optimizer=Adam(1e-5))


def get_ngram_set(x, n):
    """生成ngram合集，返回结果格式是:
    {(n-1)-gram: set([n-gram的第n个字集合])}
    """
    result = {}
    for i in range(len(x) - n + 1):
        k = tuple(x[i:i + n])
        if k[:-1] not in result:
            result[k[:-1]] = set()
        result[k[:-1]].add(k[-1])
    return result


def gen_answer(question, passage):
    """由于是MLM模型，所以可以直接argmax解码。
    """
    all_p_token_ids, token_ids, segment_ids = [], [], []
    qp_token_ids, qp_segment_ids = tokenizer.encode(question, passage, maxlen=max_qp_len + 1)
    p_token_ids, _ = tokenizer.encode(passage, maxlen=max_a_len + 1)
    all_p_token_ids.append(p_token_ids[1:])
    token_ids.append([tokenizer._token_start_id])
    token_ids[-1] += ([tokenizer._token_mask_id] * max_a_len)
    token_ids[-1] += [tokenizer._token_end_id]
    token_ids[-1] += (qp_token_ids[1:])
    segment_ids.append([1] * (max_a_len+2) + qp_segment_ids[1:])
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    probas = model.predict([token_ids, segment_ids])
    print('probas: ',probas)
    results = {}
    for t, p in zip(all_p_token_ids, probas):
        a, score = tuple(), 0.
        for i in range(max_a_len):
            idxs = list(get_ngram_set(t, i + 1)[a])
            print(idxs)
            if tokenizer._token_end_id not in idxs:
                idxs.append(tokenizer._token_end_id)
            # pi是将passage以外的token的概率置零
            # pi = np.zeros_like(p[i])
            # pi[idxs] = p[i, idxs]
            # a = a + (pi.argmax(),)
            # score += pi.max()
            if a[-1] == tokenizer._token_end_id:
                break
        score = score / (i + 1)
        a = tokenizer.decode(a)
        if a:
            results[a] = results.get(a, []) + [score]
    results = {
        k: (np.array(v)**2).sum() / (sum(v) + 1)
        for k, v in results.items()
    }
    return results


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, question, passage,topk=1):
        qp_token_ids, qp_segment_ids = tokenizer.encode(question, passage, maxlen=max_qp_len)
        output_ids = self.beam_search([qp_token_ids, qp_segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def max_in_dict(d):
    if d:
        return sorted(d.items(), key=lambda s: -s[1])[0][0]


def predict_to_file(data, filename):
    """将预测结果输出到文件，方便评估
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for d in tqdm(iter(data), desc=u'正在预测(共%s条样本)' % len(data)):
            line = d.strip().split(',')
            q_text = line[0]
            p_text = line[2]
            final_answer = line[1]
            pred_text = autotitle.generate(q_text, p_text,1)
            # a = gen_answer(q_text, p_text)
            # a = max_in_dict(a)
            print('Question: ',q_text,'True: ',final_answer,'Input :',p_text,'Prediction: ',pred_text)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./resmodel_test/best_model.weights')


if __name__ == '__main__':

   # evaluator = Evaluator()
   # train_generator = data_generator(train_data, batch_size)
   # model.fit( train_generator.forfit(),
   #     steps_per_epoch=len(train_generator),
   #     epochs=epochs,
   #     callbacks=[evaluator]
   # )

    model.load_weights('./resmodel_test/best_model.weights')
    data = data1[:50] + data2[:50]
    predict_to_file(data,'./res.csv')


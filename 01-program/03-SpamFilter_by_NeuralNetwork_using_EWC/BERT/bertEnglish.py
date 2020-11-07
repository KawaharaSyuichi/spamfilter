import pandas as pd
import numpy as np
import torch
import transformers

from transformers import BertJapaneseTokenizer
from tqdm import tqdm
tqdm.pandas()


class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(
            self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor(
            [inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        seq_out, pooled_out = self.bert_model(inputs_tensor, masks_tensor)

        if torch.cuda.is_available():
            # 0番目は [CLS] token, 768 dim の文章特徴量
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()


def cos_sim_matrix(matrix):
    """
    item-feature 行列が与えられた際に
    item 間コサイン類似度行列を求める関数
    """
    d = matrix @ matrix.T  # item-vector 同士の内積を要素とする行列

    # コサイン類似度の分母に入れるための、各 item-vector の大きさの平方根
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5

    # それぞれの item の大きさの平方根で割っている（なんだかスマート！）
    return d / norm / norm.T


if __name__ == '__main__':
    mail_path = '/Users/kawahara/documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_mail_set_by_2020/2006/2006_2000_mails'
    mail_df = pd.DataFrame([], columns=['text'])

    with open(mail_path, 'r') as f:
        mail_bodys = f.readlines()
        for i, mail_body in enumerate(mail_bodys):
            mail_df.loc[len(mail_df)] = mail_body

    BSV = BertSequenceVectorizer()

    mail_df['text_feature'] = mail_df['text'].progress_apply(
        lambda x: BSV.vectorize(x))

    mail_df.to_csv('bert_result_2005.csv', header=False,
                   index=False, columns=['text_feature'])

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

    sample_df = pd.DataFrame(['お腹が痛いので遅れます。',
                              '頭が痛いので遅れます。',
                              'おはようございます！',
                              'kaggle が好きなかえる',
                              '味噌汁が好きなワニ',
                              'From bounce-debian-mirrors=ktwarwic=speedy.uwaterloo.ca@lists.debian.org  Sun Apr  8 13:09:29 2007 Return-Path: <bounce-debian-mirrors=ktwarwic=speedy.uwaterloo.ca@lists.debian.org> Received: from murphy.debian.org (murphy.debian.org [70.103.162.31]) 	by speedy.uwaterloo.ca (8.12.8/8.12.5) with ESMTP id l38H9S0I003031 	for <ktwarwic@speedy.uwaterloo.ca>; Sun, 8 Apr 2007 13:09:28 -0400 Received: from localhost (localhost [127.0.0.1]) 	by murphy.debian.org (Postfix) with QMQP 	id 90C152E68E; Sun,  8 Apr 2007 12:09:05 -0500 (CDT) Old-Return-Path: <yan.morin@savoirfairelinux.com> X-Spam-Checker-Version: SpamAssassin 3.1.4 (2006-07-26) on murphy.debian.org X-Spam-Level:  X-Spam-Status: No, score=-1.1 required=4.0 tests=BAYES_05 autolearn=no  	version=3.1.4 X-Original-To: debian-mirrors@lists.debian.org Received: from xenon.savoirfairelinux.net (savoirfairelinux.net [199.243.85.90]) 	by murphy.debian.org (Postfix) with ESMTP id 827432E3E5 	for <debian-mirrors@lists.debian.org>; Sun,  8 Apr 2007 11:52:35 -0500 (CDT) Received: from [192.168.0.101] (bas6-montreal28-1177925679.dsl.bell.ca [70.53.184.47]) 	by xenon.savoirfairelinux.net (Postfix) with ESMTP id C1223F69B7 	for <debian-mirrors@lists.debian.org>; Sun,  8 Apr 2007 12:52:34 -0400 (EDT) Message-ID: <46191DCE.3020508@savoirfairelinux.com> Date: Sun, 08 Apr 2007 12:52:30 -0400 From: Yan Morin <yan.morin@savoirfairelinux.com> User-Agent: Icedove 1.5.0.10 (X11/20070329) MIME-Version: 1.0 To: debian-mirrors@lists.debian.org Subject: Typo in /debian/README X-Enigmail-Version: 0.94.2.0 Content-Type: text/plain; charset=ISO-8859-1 Content-Transfer-Encoding: 7bit X-Rc-Spam: 2007-01-18_01 X-Rc-Virus: 2006-10-25_01 X-Rc-Spam: 2007-01-18_01 Resent-Message-ID: <tHOiyB.A.jEC.xGSGGB@murphy> Resent-From: debian-mirrors@lists.debian.org X-Mailing-List: <debian-mirrors@lists.debian.org>  X-Loop: debian-mirrors@lists.debian.org List-Id: <debian-mirrors.lists.debian.org> List-Post: <mailto:debian-mirrors@lists.debian.org> List-Help: <mailto:debian-mirrors-request@lists.debian.org?subject=help> List-Subscribe: <mailto:debian-mirrors-request@lists.debian.org?subject=subscribe> List-Unsubscribe: <mailto:debian-mirrors-request@lists.debian.org?subject=unsubscribe> Precedence: list Resent-Sender: debian-mirrors-request@lists.debian.org Resent-Date: Sun,  8 Apr 2007 12:09:05 -0500 (CDT) Status: RO Content-Length: 729 Lines: 26  Hi,  just updated from the gulus and I check on other mirrors. It seems there is a little typo in /debian/README file  Example: http://gulus.usherbrooke.ca/debian/README ftp://ftp.fr.debian.org/debian/README  "Testing, or lenny.  Access this release through dists/testing.  The current tested development snapshot is named etch.  Packages which have been tested in unstable and passed automated tests propogate to this release."  etch should be replace by lenny like in the README.html - -  Yan Morin Consultant en logiciel libre yan.morin@savoirfairelinux.com 514-994-1556 - -  To UNSUBSCRIBE, email to debian-mirrors-REQUEST@lists.debian.org with a subject of "unsubscribe". Trouble? Contact listmaster@lists.debian.org'
                              ], columns=['text'])

    BSV = BertSequenceVectorizer()

    sample_df['text_feature'] = sample_df['text'].progress_apply(
        lambda x: BSV.vectorize(x))
    print(sample_df.head())

    print(sample_df.text_feature[5])

    print(cos_sim_matrix(np.stack(sample_df.text_feature)))

    '''
    array([[0.99999976, 0.96970403, 0.7450729 , 0.6458974 , 0.6031469 ],
           [0.96970403, 1.0000002 , 0.7233708 , 0.6374834 , 0.59641033],
           [0.7450729 , 0.7233708 , 1.0000002 , 0.6782884 , 0.62431043],
           [0.6458974 , 0.6374834 , 0.67828834, 1.0000001 , 0.8846145 ],
           [0.603147  , 0.5964103 , 0.62431043, 0.8846146 , 1.0000002 ]],
          dtype=float32)
    '''

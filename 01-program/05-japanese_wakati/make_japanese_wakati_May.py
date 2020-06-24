#import re
#nihongo = re.compile('[ぁ-んァ-ン一-龥ー0-9a-zA-Z\-()<>*。、.]+')
import random
import re
from janome.tokenizer import Tokenizer

spam_path = "/Users/kawahara/Documents/10-maildata/01-spam/spam_mails/"
spam_body = []


def split(doc, word_class=["形容詞", "形容動詞", "感動詞", "副詞", "連体詞", "名詞", "動詞"]):
    t = Tokenizer()
    tokens = t.tokenize(doc)
    word_list = []
    for token in tokens:
        word_list.append(token.surface)
    return [word for word in word_list]


def getwords(doc, doc_num):
    print("doc_num:", doc_num)
    words = [s.lower() for s in split(doc)]
    return words


with open(spam_path + "spam_at_May_wakati.txt", "a") as mail_write_path:
    with open(spam_path + "spam_at_May.txt", "r") as mail_read_path:
        spam_mails = mail_read_path.readlines()

    for mail in spam_mails:
        spam_body.append(mail)

    for num, body in enumerate(spam_body):  # spam
        words = getwords(body, num + 1)
        string = ' '.join(words)
        mail_write_path.write(string + "\n")

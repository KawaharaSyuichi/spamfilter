import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from janome.tokenizer import Tokenizer

mails_path = "MAILS PATH"

# ５月分(ham)
Mails_data = open(mails_path, 'r')

print("Finish read mails.\n")

Mails_trainings = [TaggedDocument(
    words=data.split(), tags=[i]) for i, data in enumerate(Mails_data)]

print("TaggedDocument finished\n")

model = Doc2Vec(documents=Mails_trainings, dm=1, size=300,
                windows=5, min_count=1, epochs=400, workers=4)

print("Make Doc2vec finished\n")

model.save("doc2vec.model")
Mails_data.close()

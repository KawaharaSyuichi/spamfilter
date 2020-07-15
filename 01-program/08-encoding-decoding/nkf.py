from glob import glob
from subprocess import call


files = glob(
    "/Users/kawahara/Documents/01-programming/00-大学研究/01-mailsearch/trec_doc2vec/trec2007/data/*")

for file_name in files:
    command = "nkf -w --overwrite " + file_name

    call(command.split(" "))

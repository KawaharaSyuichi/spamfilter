import csv

doc2vec_model_files = ["mail_set_1_doc2vec.csv", "mail_set_2_doc2vec.csv"]
model_doc2vec_dict = dict()
temp_vector = []
save_vector = []


for i, doc2vec_model_file in enumerate(doc2vec_model_files):
    with open("../../02-result/trec_doc2vec/trec_doc2vec_model_over_year/" + doc2vec_model_file, "r") as read_f:
        reader = csv.reader(read_f)
        temp_vector = [row for row in reader]
        """
        [直前までの累計数:必要なメール数]
        """
        if i == 0:
            # mail_set_1_spam
            save_vector.extend(temp_vector[:200])
            save_vector.extend(temp_vector[209:359])
            save_vector.extend(temp_vector[374:524])
            save_vector.extend(temp_vector[572:672])
            save_vector.extend(temp_vector[712:962])
            save_vector.extend(temp_vector[1008:1158])
            save_vector.extend(temp_vector[1273:1423])
            save_vector.extend(temp_vector[1469:1669])
            save_vector.extend(temp_vector[1829:1979])
            save_vector.extend(temp_vector[2336:2536])
            save_vector.extend(temp_vector[3011:3161])
            save_vector.extend(temp_vector[3625:3775])

            # mail_set_1_ham
            save_vector.extend(temp_vector[4149:4249])
            save_vector.extend(temp_vector[4279:4329])
            save_vector.extend(temp_vector[4362:4462])
            save_vector.extend(temp_vector[4496:4646])
            save_vector.extend(temp_vector[4703:4853])
            save_vector.extend(temp_vector[4897:4997])
            save_vector.extend(temp_vector[5045:5145])
            save_vector.extend(temp_vector[5146:5346])
            save_vector.extend(temp_vector[5371:5671])
            save_vector.extend(temp_vector[5723:6023])
            save_vector.extend(temp_vector[6109:6359])
            save_vector.extend(temp_vector[6382:6582])

            with open("../../02-result/trec_doc2vec/trec_doc2vec_model_over_year/mail_set_1_doc2vec_ver2.csv", "a") as write_f:
                writer = csv.writer(write_f, lineterminator="\n")
                writer.writerows(save_vector)

            save_vector.clear()

        elif i == 1:
            # mail_set_2_spam
            save_vector.extend(temp_vector[:32])
            save_vector.extend(temp_vector[32:41])
            save_vector.extend(temp_vector[41:47])
            save_vector.extend(temp_vector[47:55])
            save_vector.extend(temp_vector[55:66])
            save_vector.extend(temp_vector[66:74])
            save_vector.extend(temp_vector[74:78])
            save_vector.extend(temp_vector[78:130])
            save_vector.extend(temp_vector[130:162])
            save_vector.extend(temp_vector[162:234])
            save_vector.extend(temp_vector[234:1117])
            save_vector.extend(temp_vector[19035:19918])

            # mail_set_2_ham
            save_vector.extend(temp_vector[32607:32608])
            save_vector.extend(temp_vector[32608:32609])
            save_vector.extend(temp_vector[32609:32611])
            save_vector.extend(temp_vector[32611:32615])
            save_vector.extend(temp_vector[32615:33611])
            save_vector.extend(temp_vector[39823:40819])

            with open("../../02-result/trec_doc2vec/trec_doc2vec_model_over_year/mail_set_2_doc2vec_ver2.csv", "a") as write_f:
                writer = csv.writer(write_f, lineterminator="\n")
                writer.writerows(save_vector)

            save_vector.clear()

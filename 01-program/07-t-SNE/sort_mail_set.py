mail_set_list = [
    "/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_doc2vec_model_straddle_year/mail_set_1_outlier",
    "/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_doc2vec_model_straddle_year/mail_set_2_outlier"]

flag = False

for list_num, mail_set_path in enumerate(mail_set_list):
    with open(mail_set_path, "r") as mail_f:
        mail_body = mail_f.read()
        mail_body = mail_body.split()
        mail_body = [int(n) for n in mail_body]
        mail_body.sort()

    with open("mail_set_" + str(list_num + 1) + "_outlier", "a") as new_mail_f:
        for num in mail_body:
            if num > 1999 and flag == False:
                new_mail_f.write("\n")
                flag = True

            new_mail_f.write(str(num) + "\n")
    flag = False

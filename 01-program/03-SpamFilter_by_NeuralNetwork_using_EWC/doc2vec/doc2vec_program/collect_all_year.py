import random

YEARS = {"2005", "2006", "2007"}

MONTHS = {'Jan', 'Feb', 'Mar', 'Apr',
          'May', 'Jun', 'Jul', 'Aug',
          'Sep', 'Oct', 'Nov', 'Dec'}

CATEGORY = {"_spam", "_ham"}

PATH = "/Users/kawahara/Documents/01-programming/00-大学研究/02-result/trec_doc2vec/trec_mail_set_by_2020/"

for year in YEARS:
    for month in MONTHS:
        for category in CATEGORY:
            try:
                with open(PATH + year + "/" + month + category, "r") as mail_file:
                    mail_body = mail_file.read()

                with open(PATH + year + "/" + year + "_all" + category, "a") as new_mail_file:
                    new_mail_file.write(mail_body)

            except FileNotFoundError:
                pass

MAIL_CATEGORY = {"_all_spam", "_all_ham"}

for mail_category in MAIL_CATEGORY:
    for year in YEARS:
        with open(PATH + year + "/" + year + mail_category, "r") as mail_file:
            mail_lines = mail_file.readlines()
            mail_lines_100 = random.sample(mail_lines, 100)

        with open(PATH + "all_year/all_year", "a") as new_mail_file:
            for mail_line in mail_lines_100:
                new_mail_file.write(mail_line)

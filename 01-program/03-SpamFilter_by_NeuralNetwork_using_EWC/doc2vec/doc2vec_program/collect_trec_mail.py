"""
各年の各月のメールのパスを収集するプログラム
"""

import re
import os

pattern = "/.*"
repattern = re.compile(pattern)

MAIL_CLASS = {'h': 'ham', 's': 'spam'}


def check_year(mail_date_header):
    """
    YEARS = ("2005", "2006", "2007")
    for year in YEARS:
        if year in mail_date_header:
            return year
    """

    YEARS = ["1990", "1991", "1992", "1993", "1994",
             "1995", "1996", "1997", "1998", "1999",
             "2000", "2001", "2002", "2003", "2004"]

    for year in YEARS:
        if year in mail_date_header:
            return year

    return None  # 上記年以外のメールはNoneで返す


def check_month(mail_date_header):
    # [Date: Sun, 2, Jan 2005 22:58:21 -0600]の形式をまず判定する
    MONTHS = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr',
              '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug',
              '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}

    for month in MONTHS.values():
        if month in mail_date_header:
            return month

    # [Date: 1/4/2005 7:27:05 AM]の形式を次に判定する
    date_pattern = r"\d+/\d+/\d+"
    date_repattern = re.compile(date_pattern)
    date_info = date_repattern.search(mail_date_header)

    if date_info is None:
        pass
    else:
        # date_info=["1","4","2005"]([month,day,year])
        date_info = date_info.group().split("/")

        if int(date_info[0]) <= 12:
            return MONTHS[str(int(date_info[0]))]  # "01" → 1 →"1"
        else:  # [Date:2005/1/4]の場合
            return MONTHS[str(int(date_info[1]))]

    # [2005. 2.14 14:25:55]の形式を次に対応する
    date_pattern = r"\d+\. \d+\.\d+"
    date_repattern = re.compile(date_pattern)
    date_info = date_repattern.search(mail_date_header)

    if date_info is None:
        pass
    else:
        date_info = date_info.group().replace(" ", "").split(
            ".")  # date_info=["2005","2","14"]([year,month,day])
        return MONTHS[date_info[1]]

    # [2007-04-24 14:25:55]の形式を次に対応する
    date_pattern = r"\d+-\d+-\d+"
    date_repattern = re.compile(date_pattern)
    date_info = date_repattern.search(mail_date_header)

    if date_info is None:
        pass
    else:
        date_info = date_info.group().split("-")

        if int(date_info[1]) > 12:  # Date: 2007-20-06
            return MONTHS[str(int(date_info[2]))]  # "06" → 6 → "6"
        else:
            return MONTHS[str(int(date_info[1]))]

    # [Date: Pt, 02 gru 2005 17:33:42 +0100]のように、上記以外の場合(文字化けして月が分からんやつ)
    return "UNKOWN"


def collect_trec_mail_by_date(year):
    with open("../../00-data/trec_index/index_" + year, "r") as index_f:
        indexs = index_f.readlines()

    for index in indexs:
        # index[0]='h' mean ham mail , 's' mean spam mail
        mail_class = MAIL_CLASS[index[0]]
        mail_path = repattern.search(index).group()

        with open("../../00-data/trec" + year + mail_path) as read_f:
            mail_lines = read_f.readlines()

        for mail_line in mail_lines:
            if mail_line.startswith('Date:'):
                mail_year = check_year(mail_line)
                if mail_year == None:  # 年が読み取れないメールは無視
                    break

                mail_month = check_month(mail_line)

                if mail_year == "2005" or mail_year == "2006" or mail_year == "2007":
                    with open(
                            "../../02-result/trec_doc2vec/trec_mail_set_by_2020/" + mail_year + "/" + mail_month + "_" + mail_class,
                            "a") as write_f:
                        mail_content = "".join(mail_lines).replace("\n", " ")

                        write_f.write(mail_content + "\n")
                else:
                    with open(
                            "../../02-result/trec_doc2vec/trec_mail_set_by_2020/other_year/" + mail_year + "_" + mail_class,
                            "a") as write_f:
                        mail_content = "".join(mail_lines).replace("\n", " ")

                        write_f.write(mail_content + "\n")

                break

            else:
                pass


def collect_trec_mail_per_month():
    """
    先にスパムメール、後に正規のメール
    2005年6月~2006年5月:1個目
    2006年6月~2007年5月:2個目
    MONTH = { 'Jan', 'Feb', 'Mar', 'Apr',
               'May', 'Jun', 'Jul', 'Aug',
               'Sep', 'Oct', 'Nov', 'Dec'}
    """
    SPAM_MAIL_PATH_1 = ["2005/Jun_spam", "2005/Jul_spam", "2005/Aug_spam", "2005/Sep_spam",
                        "2005/Oct_spam", "2005/Nov_spam", "2005/Dec_spam", "2006/Jan_spam",
                        "2006/Feb_spam", "2006/Mar_spam", "2006/Apr_spam", "2006/May_spam"]
    HAM_MAIL_PATH_1 = ["2005/Jun_ham", "2005/Jul_ham", "2005/Aug_ham", "2005/Sep_ham",
                       "2005/Oct_ham", "2005/Nov_ham", "2005/Dec_ham", "2006/Jan_ham",
                       "2006/Feb_ham", "2006/Mar_ham", "2006/Apr_ham", "2006/May_ham"]

    SPAM_MAIL_PATH_2 = ["2006/Jun_spam", "2006/Jul_spam", "2006/Aug_spam", "2006/Sep_spam",
                        "2006/Oct_spam", "2006/Nov_spam", "2006/Dec_spam", "2007/Jan_spam",
                        "2007/Feb_spam", "2007/Mar_spam", "2007/Apr_spam", "2007/May_spam"]
    HAM_MAIL_PATH_2 = ["2006/Jun_ham", "2006/Jul_ham", "2006/Aug_ham", "2006/Sep_ham",
                       "2006/Oct_ham", "2006/Nov_ham", "2006/Dec_ham", "2007/Jan_ham",
                       "2007/Feb_ham", "2007/Mar_ham", "2007/Apr_ham", "2007/May_ham"]

    for mail_path in SPAM_MAIL_PATH_1:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.read()

            with open("../trec_mail_set_by_2020/collect_mail_set/mail_set_1", "a") as f:
                f.write(mail_body)
        except:
            pass

    for mail_path in HAM_MAIL_PATH_1:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.read()

            with open("../trec_mail_set_by_2020/collect_mail_set/mail_set_1", "a") as f:
                f.write(mail_body)
        except:
            pass

    for mail_path in SPAM_MAIL_PATH_2:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.read()

            with open("../trec_mail_set_by_2020/collect_mail_set/mail_set_2", "a") as f:
                f.write(mail_body)
        except:
            pass

    for mail_path in HAM_MAIL_PATH_2:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.read()

            with open("../trec_mail_set_by_2020/collect_mail_set/mail_set_2", "a") as f:
                f.write(mail_body)
        except:
            pass


def count_mail_num():
    mail_set_1_spam_num = 0
    mail_set_1_ham_num = 0
    mail_set_2_spam_num = 0
    mail_set_2_ham_num = 0

    SPAM_MAIL_PATH_1 = ["2005/Jun_spam", "2005/Jul_spam", "2005/Aug_spam", "2005/Sep_spam",
                        "2005/Oct_spam", "2005/Nov_spam", "2005/Dec_spam", "2006/Jan_spam",
                        "2006/Feb_spam", "2006/Mar_spam", "2006/Apr_spam", "2006/May_spam", ]
    HAM_MAIL_PATH_1 = ["2005/Jun_ham", "2005/Jul_ham", "2005/Aug_ham", "2005/Sep_ham",
                       "2005/Oct_ham", "2005/Nov_ham", "2005/Dec_ham", "2006/Jan_ham",
                       "2006/Feb_ham", "2006/Mar_ham", "2006/Apr_ham", "2006/May_ham", ]

    SPAM_MAIL_PATH_2 = ["2006/Jun_spam", "2006/Jul_spam", "2006/Aug_spam", "2006/Sep_spam",
                        "2006/Oct_spam", "2006/Nov_spam", "2006/Dec_spam", "2007/Jan_spam",
                        "2007/Feb_spam", "2007/Mar_spam", "2007/Apr_spam", "2007/May_spam", ]
    HAM_MAIL_PATH_2 = ["2006/Jun_ham", "2006/Jul_ham", "2006/Aug_ham", "2006/Sep_ham",
                       "2006/Oct_ham", "2006/Nov_ham", "2006/Dec_ham", "2007/Jan_ham",
                       "2007/Feb_ham", "2007/Mar_ham", "2007/Apr_ham", "2007/May_ham", ]

    for mail_path in SPAM_MAIL_PATH_1:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.readlines()
                print(mail_path + ":", len(mail_body))
                mail_set_1_spam_num += len(mail_body)
        except:
            pass

    for mail_path in HAM_MAIL_PATH_1:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.readlines()
                print(mail_path + ":", len(mail_body))
                mail_set_1_ham_num += len(mail_body)
        except:
            pass

    for mail_path in SPAM_MAIL_PATH_2:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.readlines()
                print(mail_path + ":", len(mail_body))
                mail_set_2_spam_num += len(mail_body)
        except:
            pass

    for mail_path in HAM_MAIL_PATH_2:
        try:
            with open("../trec_mail_set_by_2020/" + mail_path, "r") as f:
                mail_body = f.readlines()
                print(mail_path + ":", len(mail_body))
                mail_set_2_ham_num += len(mail_body)
        except:
            pass

    print("mail_set_1_spam_num", mail_set_1_spam_num)
    print("mail_set_1_ham_num", mail_set_1_ham_num)
    print("mail_set_2_spam_num", mail_set_2_spam_num)
    print("mail_set_2_ham_num", mail_set_2_ham_num)


if __name__ == "__main__":

    years = ["2005", "2006", "2007"]

    for year in years:
        collect_trec_mail_by_date(year)

    """
    # collect_trec_mail_per_month()
    count_mail_num()
    """

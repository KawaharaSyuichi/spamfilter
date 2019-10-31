import imaplib
import email
import datetime
import something_info  # メールサーバのパスワード等の情報を記載したファイル

from email.header import make_header, decode_header

docomo = imaplib.IMAP4_SSL(something_info.SERVER_NAME, 993)
docomo.login(something_info.USER_ID, something_info.PASSWORD)
docomo.select('inbox')

month_dict = {1: ["Jan", 31], 2: ["Feb", 28, 29], 3: ["Mar", 31], 4: ["Apr", 30], 5: ["May", 31], 6: ["Jun", 30],
              7: ["Jul", 31], 8: ["Aug", 31], 9: ["Sep", 30], 10: ["Oct", 31], 11: ["Nov", 30], 12: ["Dec", 31]}


def isLeapYear(year):
    return (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))


def make_yesterday_str():
    date = datetime.date.today()
    today = str(date.day) + '-' + \
        month_dict[date.month][0] + '-' + str(date.year)

    if (date.today - 1) == 0:  # 月初
        if (date.month == 3 and isLeapYear(data.year)):  # 前日が2月かつ閏年の場合
            yesterday = str(month_dict[2][2]) + '-' + \
                month_dict[date.month - 1][0] + '-' + str(date.year)

        elif (date.month == 3):  # 前日が2月かつ閏年でない場合
            yesterday = str(month_dict[2][1]) + '-' + \
                month_dict[date.month - 1][0] + '-' + str(date.year)

        elif (date.month == 1):  # 前日が12月の場合
            yesterday = str(month_dict[12][1]) + '-' + \
                month_dict[12][0] + '-' + str(date.year - 1)

        else:
            yesterday = str(month_dict[date.month-1][1]) + '-' + \
                month_dict[date.month-1][0] + '-' + str(date.year)

    else:
        yesterday = str(date.day - 1) + '-' + \
            month_dict[date.month] + '-' + str(date.year)

    search_option = 'SINCE ' + yesterday + ' BEFORE ' + today

    return yesterday, search_option


def write_mailtext(msg, yesterday, skip_mail_num):

    try:
        from_addr = str(make_header(decode_header(msg["From"])))
    except:
        skip_mail_num += 1
        return

    if from_addr in something_info.notspam_address_list:  # スパムではないアドレスの確認
        # スパムでないアドレスが確認できた場合の処理
        skip_mail_num += 1
        return

    with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
        for mail_header in msg.keys():
            try:
                header_document = str(make_header(
                    decode_header(msg[mail_header])))

                header_document = header_document.replace('\r\n', '<br>')
                header_document = header_document.replace('\n', '<br>')
                header_document = header_document.replace('\r', '<br>')

                spam.write(mail_header+":" + header_document+"\n")
            except:
                spam.write(mail_header+":" + "MaybeUnicodeDecodeError"+"\n")

    # 本文(payload)を取得する
    if msg.is_multipart() is False:
        # シングルパートのとき
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset()

        if charset is not None:
            payload = payload.decode(charset, "ignore")

        try:
            payload = payload.replace('\r\n', '')
            payload = payload.replace('\n', '')
            payload = payload.replace('\r', '')
            with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                spam.write("body:" + payload + '\n' * 2)
        except:
            with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                spam.write("body:" + "Decode_Error" + '\n' * 2)

    else:
        # マルチパートのとき
        skip_flag = False
        for part in msg.walk():
            payload = part.get_payload(decode=True)

            if payload is None:
                continue

            charset = part.get_content_charset()

            if charset is not None:
                payload = payload.decode(charset, "ignore")

            try:
                payload = payload.replace('\r\n', '')
                payload = payload.replace('\n', '')
                payload = payload.replace('\r', '')
                if skip_flag == False:
                    skip_flag = True
                    with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                        spam.write("body:" + payload)
                else:
                    with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                        spam.write(payload)
            except:
                if skip_flag == False:
                    skip_flag = True
                    with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                        spam.write("body:" + "Decode_Error")
                else:
                    with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
                        spam.write("Decode_Error")

        with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
            spam.write("\n"*2)


# 取得したメール一覧の処理
skip_mail_num = 0
yesterday, search_option = make_yesterday_str()

_, datas = docomo.search(None, search_option)

for i, num in enumerate(datas[0].split()):
    _, data = docomo.fetch(num, '(RFC822)')

    try:
        msg = email.message_from_bytes(data[0][1])
    except:
        skip_mail_num += 1
        continue

    write_mailtext(msg, yesterday, skip_mail_num)

    print(i+1)

with open(something_info.spam_path + "spam_" + yesterday + ".txt", "a") as spam:
    spam.write("Total spam mail num:" +
               str(len(datas[0].split())-skip_mail_num))

docomo.close()
docomo.logout()

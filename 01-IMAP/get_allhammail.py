import imaplib
import email
import datetime
import ham_info

from email.header import make_header, decode_header

UserName = ham_info.USER_ID
PassName = ham_info.PASSWORD

gmail = imaplib.IMAP4_SSL("imap.gmail.com", '993')
gmail.login(UserName, PassName)
gmail.select("inbox")

month_dict = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

date = datetime.date.today()
today = str(date.day) + '-' + month_dict[date.month] + '-' + str(date.year)
yesterday = str(date.day - 1) + '-' + \
    month_dict[date.month] + '-' + str(date.year)

search_option = 'SINCE ' + yesterday + ' BEFORE ' + today
_, datas = gmail.search(None, "ALL")


def write_mailtext(msg, skip_mail_num):

    with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
        for mail_header in msg.keys():
            try:
                header_document = str(make_header(
                    decode_header(msg[mail_header])))

                header_document = header_document.replace('\r\n', '<br>')
                header_document = header_document.replace('\n', '<br>')
                header_document = header_document.replace('\r', '<br>')

                ham.write(mail_header+":" + header_document+"\n")
            except:
                ham.write(mail_header+":" + "MaybeUnicodeDecodeError"+"\n")

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
            with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                ham.write("body:" + payload + '\n' * 2)
        except:
            with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                ham.write("body:" + "Decode_Error" + '\n' * 2)

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
                    with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                        ham.write("body:" + payload)
                else:
                    with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                        ham.write(payload)
            except:
                if skip_flag == False:
                    skip_flag = True
                    with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                        ham.write("body:" + "Decode_Error")
                else:
                    with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
                        ham.write("Decode_Error")
        with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
            ham.write("\n"*2)


# 取得したメール一覧の処理
skip_mail_num = 0

for num in datas[0].split():
    _, data = gmail.fetch(num, '(RFC822)')

    try:
        msg = email.message_from_bytes(data[0][1])
    except:
        skip_mail_num += 1

    write_mailtext(msg, skip_mail_num)

with open(ham_info.ham_path + "ham_all.txt", "a") as ham:
    ham.write("Total ham mail num:" +
              str(len(datas[0].split())-skip_mail_num)+"\n")

gmail.close()
gmail.logout()

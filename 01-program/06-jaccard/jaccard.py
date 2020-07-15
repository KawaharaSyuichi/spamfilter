# Jaccard係数(overlap coefficient)の計算アルゴリズム
def jaccard_similarity_coefficient(list_a, list_b):
    # 集合Aと集合Bの積集合(set型)を作成
    set_intersection = set.intersection(set(list_a), set(list_b))
    # 集合Aと集合Bの積集合の要素数を取得
    num_intersection = len(set_intersection)

    # 集合Aと集合Bの和集合(set型)を作成
    set_union = set.union(set(list_a), set(list_b))
    # 集合Aと集合Bの和集合の要素数を取得
    num_union = len(set_union)

    # 積集合の要素数を和集合の要素数で割って
    # Jaccard係数を算出
    try:
        return float(num_intersection) / num_union
    except ZeroDivisionError:
        return 1.0


def all_mail_jaccard(years, mail_path, mail_category):
    year_1_mail_bodies = str()
    year_2_mail_bodies = str()

    for i, year in enumerate(years):
        with open(mail_path + str(year) + '/' + str(year) + '_2000_mails',
                  'r') as f:
            mail_bodies = f.readlines()

        if mail_category == 'all':
            for mail_body in mail_bodies:
                if i == 0:
                    year_1_mail_bodies += ' '.join(mail_body)
                else:
                    year_2_mail_bodies += ' '.join(mail_body)

        elif mail_category == 'spam':
            for n, mail_body in enumerate(mail_bodies):
                if i == 0:
                    year_1_mail_bodies += ' '.join(mail_body)
                else:
                    year_2_mail_bodies += ' '.join(mail_body)

                if n == 999:
                    break

        elif mail_category == 'ham':
            for m, mail_body in enumerate(mail_bodies):
                if m <= 999:
                    continue

                if i == 0:
                    year_1_mail_bodies += ' '.join(mail_body)
                else:
                    year_2_mail_bodies += ' '.join(mail_body)

    jaccard_result = jaccard_similarity_coefficient(year_1_mail_bodies.split(),
                                                    year_2_mail_bodies.split())

    return jaccard_result


def main():
    COMMON_PATH = '/Users/kawahara/Documents/01-programming/00-大学研究/02-result' \
                  '/trec_doc2vec/trec_mail_set_by_2020/'

    mail_categories_list = ['all', 'spam', 'ham']

    for mail_category in mail_categories_list:
        print("=" * 15 + mail_category + "=" * 15)

        jaccard = all_mail_jaccard([2005, 2006], COMMON_PATH, mail_category)
        print('Jaccard of 2005 and 2006 is :{}'.format(jaccard))
        jaccard = all_mail_jaccard([2005, 2007], COMMON_PATH, mail_category)
        print('Jaccard of 2005 and 2007 is :{}'.format(jaccard))
        jaccard = all_mail_jaccard([2006, 2007], COMMON_PATH, mail_category)
        print('Jaccard of 2006 and 2007 is :{}'.format(jaccard))

        print("=" * 30)


if __name__ == '__main__':
    main()

import argparse
from statsmodels.stats.contingency_tables import mcnemar


def main(args):
    mcnemar_test(args.contingency)


def mcnemar_test(contingency):
    yesyes = contingency[0]
    yesno = contingency[1]
    noyes = contingency[2]
    nono = contingency[3]
    print('yes/yes: {}, yes/no: {}, no/yes: {}, no/no: {}'.format(yesyes, yesno, noyes, nono))
    table = [[yesyes, yesno], [noyes, nono]]
    result = mcnemar(table, exact=False, correction=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--contingency',
                        nargs='+',
                        help='yesyes, yesno, noyes, nono',
                        type=str)

    args = parser.parse_args()
    main(args)
import ipdb

def cal_seen_unseen_stats(preds, gts):

    unseen = set([5, 11, 12])

    unseen_total = unseen_match = 0
    seen_total = seen_match = 0
    for i in range(len(gts)):
        if gts[i] in unseen:
            unseen_total += 1
            if preds[i] == gts[i]:
                unseen_match += 1
        else:
            seen_total += 1
            if preds[i] == gts[i]:
                seen_match += 1
    #已见类别的样本总数
    print(f"seen_total: {seen_total}")
    #已见类别中正确预测的样本数量
    print(f"seen_match: {seen_match}")
    #已见类别中正确预测的概率
    print(seen_match / seen_total)
    #未见类别中样本总数和正确预测数量
    print(f"unseen_total: {unseen_total}")
    print(f"unseen_match: {unseen_match}")
    if(unseen_total != 0):
        print(unseen_match / unseen_total)



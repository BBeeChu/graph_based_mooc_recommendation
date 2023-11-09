def ndcg(rate, negative, length, k=5):
    negative = negative.long()
    # rate에서 negative 인덱스에 해당하는 값을 가져온다.
    test = rate[negative[:, :, 0], negative[:, :, 1]]

    # 상위 k개의 인덱스를 구한다.
    topk_values, topk_indices = torch.topk(test, k=k, dim=1, largest=True, sorted=True)

    # 99와 일치하는 인덱스를 찾는다.
    n = (topk_indices == 99).nonzero(as_tuple=False)[:, 1]

    # NDCG를 계산한다.
    ndcg_score = torch.sum(torch.log2(torch.tensor(2.0)) / torch.log2(n.to(torch.float32) + 2)) / length

    return ndcg_score
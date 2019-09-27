import torch


def GramMatrix(images):
    n, d = images.shape[0:2]
    images = images.view(n, d, -1)
    return images.bmm(images.transpose(1,2))


def GramMatrices(layers):
    tmp = []
    for l in layers:
        tmp.append(GramMatrix(l))
    return tmp


def euclidian_dist(features1, features2, p=1):
    return torch.dist(features1,features2, p=p)


def content_loss(f1, f2, weights):
    assert len(f1)==len(f2)==len(weights), "len of input should be the same."
    loss = 0.0
    for i in range(len(weights)):
        loss = loss + euclidian_dist(f1[i], f2[i]) * weights[i]

    return loss


def style_loss(f1, gram_mean2, weights):
    assert len(f1)==len(weights), "len of input should be the same."
    loss = 0.0
    for i in range(len(weights)):
        loss = loss + euclidian_dist(GramMatrix(f1[i]), gram_mean2[i]) * weights[i]

    return loss

import torch

class AvgrageMeter(object):
    #描述目标检测的正确率的类
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        #输入的val是batch中的正确率
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, label, topk=(1,)):
    '''
    计算准确率
    '''
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)  #使用topk获得前k个索引

    pred = pred.t()     #进行转置
    correct = pred.eq(label.view(1, -1).expand_as(pred))    #与label进行比较
    #eq按照对应元素进行比较
    #view(1,-1)自动转换为行为1的形状
    #expand_as(pred)将label的标签扩展到pred的shape

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0) #前k行数据，平整到1维度，计算true的个数
        res.append(correct_k.mul_(100.0 / batch_size))  #mul_乘法，变成百分比，即为准确率
    return res

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval() # 评估模式, 这会关闭dropout
            #计算一个batch中预测对的个数
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train() # 改回训练模式
            # 累加上batch size的个数
            n += y.shape[0]
    return acc_sum / n
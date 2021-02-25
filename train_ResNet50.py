from train import *
from torchvision import transforms

'''
训练ResNet50并且进行数据增强，并每epoch保存一次模型
每个epoch保存一次
'''


def train_get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 20  # 训练20轮
    net = ResNet.ResNet50()
    train_path = C.train_path
    test_path = C.test_path
    tensorboard_path = C.tensorboard_path

    # 数据增强
    train_transform = transforms.Compose([

        transforms.Resize(size= (256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),      #水平翻转
        transforms.RandomRotation(15),          #随机旋转,-15~15范围内
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 读取数据
    train_ds = MyDataset(train_path, transform=train_transform)
    new_train_ds, validate_ds = dataset_split(train_ds, 0.96)  # 留1000张图片作为validation_dataset
    test_ds = MyDataset(test_path, train=False)

    train_loader = dataloader(train_ds)
    new_train_loader = dataloader(new_train_ds)
    validate_loader = dataloader(validate_ds)
    test_loader = dataloader(test_ds)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(epochs, new_train_loader, device, net, criterion, optimizer, tensorboard_path, validate_loader)


def make_csv():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置路径
    csv_path = C.csv_path
    test_path = C.test_path
    # 读取数据
    test_ds = MyDataset(test_path, train=False)
    test_loader = dataloader(test_ds)
    # 加载网络
    net = ResNet.ResNet50()
    net.load_state_dict(torch.load(C.model_save_path + 'ResNet-18-epoch:19.pth'))
    # print(net)  #测试是否读取成功
    print('model loaded successfully!')

    # 测试结果，写csv文件
    submission(csv_path=C.csv_path, test_loader=test_loader, device=device, model=net)


if __name__ == '__main__':
    train_get_model()  # 训练并存储模型参数（每个epoch存储一次）
    # make_csv()  #加载已有模型预测测试集并写入csv
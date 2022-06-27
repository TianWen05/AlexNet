import time
import torch
import torchvision
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *


def main():
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    print("训练于{}上".format(device))
    print("{}线程工作".format(num_workers))

    # 准备数据集
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=224),
        torchvision.transforms.ToTensor()
    ])
    train_data = torchvision.datasets.FashionMNIST(root="./Datasets", train=True, transform=dataset_transform,
                                                   download=True)
    test_data = torchvision.datasets.FashionMNIST(root="./Datasets", train=False, transform=dataset_transform,
                                                  download=True)
    # 获取数据长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    # 查看数据长度
    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    # 利用DataLoader来进行加载
    train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=num_workers)

    # 创建网络模型
    net = AlexNet()
    net = net.to(device)

    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # 设置训练网络的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 10

    # 添加tensorboard
    writer = SummaryWriter("./logs_train")
    start_time = time.time()

    # 删除模型日志文件
    del_files("./logs_train")
    del_files("./model")

    for i in range(epoch):
        print("---------------------第 {} 轮训练开始---------------------".format(i+1))

        # 训练步骤开始
        net.train()
        train_start_time = time.time()
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)
            loss = loss_fn(output, targets)

            # 优化器优化模型
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器优化参数
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                now_time = time.time()
                print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
                print("目前时间为：{}\n".format(now_time - train_start_time))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        train_end_time = time.time()
        print("该轮训练花费时间：{}\n".format(train_end_time - train_start_time))

        # 测试步骤开始
        net.eval()
        total_test_loss = 0
        total_test_acc = 0
        test_start_time = time.time()
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = net(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_test_acc += accuracy
        test_end_time = time.time()
        print("该轮测试花费时间：{}".format(test_end_time - test_start_time))
        print("整体测试集上的loss：{}".format(total_test_loss))
        print("整体测试集的正确率：{}".format(total_test_acc/test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_test_acc/test_data_size, total_test_step)
        total_test_step += 1

        torch.save(net, "./model/AlexNet_model_{}.pth".format(i+1))
        print("模型已保存")
    end_time = time.time()
    print("总训练花费时间：{}".format(end_time - start_time))

    writer.close()


def del_files(path_file):
    ls = os.listdir(path_file)
    for i in ls:
        f_path = os.path.join(path_file, i)
        # 判断是否是一个目录,若是,则递归删除
        if os.path.isdir(f_path):
            del_files(f_path)
        else:
            os.remove(f_path)


if __name__ == '__main__':
    main()

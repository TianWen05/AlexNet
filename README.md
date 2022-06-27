# 说明
## 文件夹
### Datasets:
    训练模型所用的数据集
### logs_train:
    模型训练时存放的日志信息，用Tensorboard打开
### model:
    存放模型训练时的模型

## 文件
### AlexNet:
    模型训练主函数模块
### model:
    模型搭建模块
### model_test:
    模型测试模块，对已训练模型来进行图片预测，直接运行即可看到效果，
    用的是test_loss最小的模型，从数据集中随机抽取一个图片来预测。

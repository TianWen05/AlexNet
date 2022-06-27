import matplotlib.pyplot as plt
import torch
import torchvision
import os
import random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = torch.load("./已训练模型和日志/model/AlexNet_model_10.pth")
label = ["T恤", "裤子", "套衫", "裙子", "外套", "凉鞋", "汗衫", "运动鞋", "包", "裸靴"]

# 加载数据
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=224),
    torchvision.transforms.ToTensor()
])
test_data = torchvision.datasets.FashionMNIST(root="./Datasets", train=False, transform=dataset_transform,
                                              download=True)
test_data_size = len(test_data)

num_rd = random.randint(0, test_data_size)
# 获取真实和预测标签
img, true_target = test_data[num_rd]
img = torch.unsqueeze(img, dim=0)
img = img.to(device)
output = model(img)
predict_target = output.argmax(1)

print("真实值为：{}".format(label[true_target]))
print("预测值为：{}".format(label[predict_target]))

img = img.cpu().clone()
img = img.numpy()[0][0]

plt.figure()
plt.imshow(img, cmap="gray")
plt.title('image')
plt.xticks([]), plt.yticks([])
plt.show()

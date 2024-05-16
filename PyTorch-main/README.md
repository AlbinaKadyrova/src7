
# Обучение нейронной сети для распознавания рукописных цифр с использованием PyTorch

Этот репозиторий содержит пример обучения нейронной сети с использованием библиотеки PyTorch для распознавания рукописных цифр из набора данных MNIST.

## Описание

Этот код демонстрирует следующие шаги:

1. Загрузка данных из набора данных MNIST.
2. Определение архитектуры нейронной сети.
3. Определение функции потерь и оптимизатора для обучения сети.
4. Обучение нейронной сети на тренировочном наборе данных.
5. Тестирование обученной сети на тестовом наборе данных.


# Определение архитектуры нейронной сети
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Обучение нейронной сети
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Обучение завершено')

# Тестирование нейронной сети
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Точность сети на 10000 тестовых изображений: %d %%' % (100 * correct / total))



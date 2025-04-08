# 1. 导入数据

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape)

# 2. 构建网络
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# 3. 训练之前，确定好损失函数、优化器、监控指标
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 训练之前，将所有输入数据转化为tensor张量
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 5. 对标签继续分类编码
from keras.utils import to_categorical 

train_labels = to_categorical(train_labels) 
test_labels = to_categorical(test_labels)

# 6. 开始训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 7. 应用训练好的网络
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)
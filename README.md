## 项目文件

```shell
.
├── choose_param.py        // 模型调参
├── compare_optimizer.py   // 优化器对比
├── data
│   ├── data_handler.py    // 数据处理文件
│   ├── test.csv
│   └── train.csv
├── imgs
├── kaggle.ipynb           // 在kaggle平台上的运行结果
├── main.py                // transormer的训练与结果展示
├── model
│   ├── lstm.py            // 基于rnn的seq2seq实现
│   └── transformer.py     // transformer的实现
├── output
│   ├── choose_param.json  // 所有超参数组合的结果
│   ├── model.pt
│   ├── result.csv         // 使用最佳参数为测试集生成的结果
│   └── nltk_data          // 计算meteor时所需要的依赖
├── README.md              // 本文件
├── REPORT.md              // 实验报告
├── REPORT.pdf             // 实验报告
├── requirements.txt       // 依赖包
├── test.py                // 训练基于rnn的seq2seq模型
└── train.py               // 模型的训练、评估定义
```

- 实验报告见REPORT.md/REPORT.pdf
- 结果文件见output/及kaggle.ipynb

## 结果复现

通过requirements.txt安装对应包:

```shell
pip install -r requirements.txt
```

由于本地设备原因，报告中的结果展示均在kaggle平台上提供的jupyter环境（使用`GPU P100`）中训练，所有的结果均可在kaggle.ipynb中查看

当然在本地同样也可以复现（需要注意batchsize的大小，默认为128，如果使用GPU训练，需要显存至少为4G，否则需要调小batchsize，这会对结果造成影响）：

RNN结构：

- 运行lstm的训练(初步模型)：

```shell
python test.py
```

Transformer结构：

- 对模型进行调参（调参需要的时间很长）：

```shell
python choose_param.py
```

- 三个优化器的对比：

```shell
python compare_optimizer.py
```

- 训练模型并进行结果展示(仅main.py可以传递参数)：

```shell
python main.py
```

- 使用最佳参数训练模型并进行结果展示

```shell
python main.py --learning_rate 0.0005 \
               --hidden_dim 256 \
               --feed_forward_dim=1024 \
               --layers 2 \
               --dropout 0.2 \
               --batch_size=128
```

各参数说明如下

```shell
options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size, defaul=128
  --max_epochs MAX_EPOCHS
                        max number of epochs, default=30
  --learning_rate LEARNING_RATE
                        learning rate, default=0.001
  --dropout DROPOUT     dropout rate, default=0.5
  --optimizer OPTIMIZER
                        optimizer type, default=adam
  --hidden_dim HIDDEN_DIM
                        hidden dim, default=128
  --feed_forward_dim FEED_FORWARD_DIM
                        feed forward dim, default=512
  --heads HEADS         heads of multi-head attention, default=8
  --layers LAYERS       layers of encoder and decoder, default=2
```

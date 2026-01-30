"""
====================================================================================
DeepFM - 深度因子分解机
====================================================================================
创建日期: July 31, 2020
更新日期: Nov 14, 2021
参考论文: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", IJCAI, 2017
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

【模型简介】
DeepFM是一种结合了FM（因子分解机）和DNN（深度神经网络）的模型。
它能够同时学习低阶和高阶的特征交叉，是工业界广泛使用的CTR预估模型。

【核心创新点】
1. FM部分学习一阶和二阶特征交叉（低阶）
2. DNN部分学习高阶特征交叉
3. 两部分共享相同的Embedding，减少参数量
4. 端到端训练，无需特征工程

【与Wide&Deep的对比】
┌─────────────────────┬────────────────────────┐
│    Wide&Deep        │       DeepFM           │
├─────────────────────┼────────────────────────┤
│ Wide部分需要手动特征  │ FM部分自动特征交叉      │
│ 工程设计交叉特征      │ 无需人工干预            │
├─────────────────────┼────────────────────────┤
│ Wide和Deep的输入不同  │ FM和Deep共享Embedding   │
├─────────────────────┼────────────────────────┤
│ 依赖专家经验          │ 端到端学习              │
└─────────────────────┴────────────────────────┘

【模型结构图】

                        输入特征
            [user_id, item_id, category, ...]
                           │
                           ↓
            ┌─────────────────────────────┐
            │        Embedding Layer       │
            │  将稀疏特征转换为稠密向量       │
            │  所有特征共享这一层            │
            └─────────────────────────────┘
                     │           │
          ┌──────────┘           └──────────┐
          ↓                                 ↓
    ┌───────────┐                   ┌───────────┐
    │  FM 组件   │                   │ DNN 组件  │
    │           │                   │           │
    │ ┌───────┐ │                   │ ┌───────┐ │
    │ │一阶线性│ │                   │ │ Dense │ │
    │ └───────┘ │                   │ └───────┘ │
    │     +     │                   │     ↓     │
    │ ┌───────┐ │                   │ ┌───────┐ │
    │ │二阶交叉│ │                   │ │ Dense │ │
    │ └───────┘ │                   │ └───────┘ │
    └─────┬─────┘                   │     ↓     │
          │                         │ ┌───────┐ │
          │                         │ │ Dense │ │
          │                         │ └───────┘ │
          │                         └─────┬─────┘
          │                               │
          └───────────→ + ←───────────────┘
                        │
                        ↓
                   sigmoid(y)
                        │
                        ↓
                  点击概率 [0, 1]

【输入数据格式】
{
    'user_id': [1, 2, 3, ...],      # 用户ID
    'item_id': [10, 20, 30, ...],   # 物品ID
    'category': [1, 2, 1, ...],     # 类别ID
    ...
}

【关键公式】
y = sigmoid(y_FM + y_DNN)
y_FM = w0 + Σ wi*xi + Σ Σ <vi, vj>*xi*xj
y_DNN = MLP(concat(e1, e2, ..., en))
====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer

from reclearn.layers import New_FM, MLP
from reclearn.layers.utils import index_mapping


class DeepFM(Model):
    """
    DeepFM模型类
    
    DeepFM是工业界最常用的CTR预估模型之一，其核心优势是：
    1. 自动学习特征交叉，无需手动特征工程
    2. 同时捕获低阶和高阶特征交互
    3. FM和DNN共享Embedding，参数效率高
    """
    
    def __init__(self, feature_columns, hidden_units=(200, 200, 200), activation='relu',
                 dnn_dropout=0., fm_w_reg=0., embed_reg=0.):
        """
        初始化DeepFM模型
        
        参数说明:
        ---------
        feature_columns : list
            特征列配置列表，每个元素是一个字典：
            [
                {'feat_name': 'user_id', 'feat_num': 10000, 'embed_dim': 8},
                {'feat_name': 'item_id', 'feat_num': 50000, 'embed_dim': 8},
                ...
            ]
            ⚠️ 注意: 所有特征的embed_dim必须相同！
            
        hidden_units : tuple, 默认(200, 200, 200)
            DNN部分的隐藏层单元数
            例如: (256, 128, 64) 表示3层，单元数分别为256、128、64
            
        activation : str, 默认'relu'
            DNN的激活函数
            可选: 'relu', 'sigmoid', 'tanh'等
            
        dnn_dropout : float, 默认0.0
            DNN的Dropout比例（0-1之间）
            用于防止过拟合
            
        fm_w_reg : float, 默认0.0
            FM一阶权重的L2正则化系数
            
        embed_reg : float, 默认0.0
            Embedding层的L2正则化系数
        """
        super(DeepFM, self).__init__()
        
        # 保存特征列配置
        self.feature_columns = feature_columns
        
        # ===== Embedding层 =====
        # 为每个特征创建独立的Embedding层
        # 这些Embedding会被FM和DNN共享
        self.embed_layers = {
            feat['feat_name']: Embedding(
                input_dim=feat['feat_num'],      # 该特征的取值个数
                input_length=1,
                output_dim=feat['embed_dim'],    # 嵌入维度
                embeddings_initializer='random_normal',
                embeddings_regularizer=l2(embed_reg)
            )
            for feat in self.feature_columns
        }
        
        # ===== 特征索引映射 =====
        # 用于将各个特征映射到全局索引空间
        # 例如: user_id的范围[0, 10000)，item_id的范围[10000, 60000)
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        
        # 特征域数量（有多少个特征）
        self.field_num = len(self.feature_columns)
        
        # 嵌入维度（假设所有特征的embed_dim相同）
        self.embed_dim = self.feature_columns[0]['embed_dim']
        
        # ===== FM层 =====
        # New_FM是专门为DeepFM设计的FM层
        # 它接收稀疏输入和嵌入输入，计算一阶和二阶得分
        self.fm = New_FM(
            feature_length=self.feature_length,  # 总特征数
            w_reg=fm_w_reg                       # 一阶权重正则化
        )
        
        # ===== DNN层 =====
        # 多层感知机，学习高阶特征交叉
        self.mlp = MLP(
            hidden_units=hidden_units,  # 隐藏层配置
            activation=activation,       # 激活函数
            dnn_dropout=dnn_dropout      # Dropout
        )
        
        # DNN的输出层
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        """
        模型前向传播
        
        参数:
        -----
        inputs : dict
            输入特征字典，键是特征名，值是特征值的batch
            例如:
            {
                'user_id': [1, 2, 3, ...],      # shape: (batch_size,)
                'item_id': [10, 20, 30, ...],   # shape: (batch_size,)
            }
        
        返回:
        -----
        outputs : Tensor
            预测的点击概率，shape: (batch_size, 1)
        
        处理流程:
        ---------
        1. 将所有特征通过Embedding层得到稠密向量
        2. FM部分: 计算一阶和二阶特征交叉得分
        3. DNN部分: 将Embedding拼接后送入MLP
        4. 将FM和DNN的输出相加，通过Sigmoid得到概率
        """
        
        # ==================== Step 1: Embedding层 ====================
        
        # 对每个特征进行Embedding查表，然后拼接
        # 例如: user_id -> (batch_size, embed_dim)
        #       item_id -> (batch_size, embed_dim)
        #       拼接后 -> (batch_size, embed_dim * field_num)
        sparse_embed = tf.concat(
            [self.embed_layers[feat_name](value) 
             for feat_name, value in inputs.items()],
            axis=-1
        )  # shape: (batch_size, embed_dim * field_num)
        
        # ==================== Step 2: FM部分 ====================
        
        # 2.1 准备FM的输入
        # FM需要知道每个特征的全局索引（用于查找一阶权重）
        sparse_inputs = index_mapping(inputs, self.map_dict)
        
        # 将Embedding reshape成 (batch_size, field_num, embed_dim)
        # 这样FM层可以分别处理每个特征域
        wide_inputs = {
            'sparse_inputs': sparse_inputs,
            'embed_inputs': tf.reshape(
                sparse_embed, 
                shape=(-1, self.field_num, self.embed_dim)
            )
        }
        
        # 2.2 计算FM得分
        # FM层返回一阶部分 + 二阶交叉部分的和
        wide_outputs = tf.reshape(
            self.fm(wide_inputs), 
            [-1, 1]
        )  # shape: (batch_size, 1)
        
        # ==================== Step 3: DNN部分 ====================
        
        # 3.1 将拼接的Embedding送入MLP
        # 输入: (batch_size, embed_dim * field_num)
        deep_outputs = self.mlp(sparse_embed)
        
        # 3.2 DNN的输出层
        deep_outputs = tf.reshape(
            self.dense(deep_outputs), 
            [-1, 1]
        )  # shape: (batch_size, 1)
        
        # ==================== Step 4: 组合输出 ====================
        
        # FM输出 + DNN输出，然后通过Sigmoid
        # y = sigmoid(y_FM + y_DNN)
        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        # shape: (batch_size, 1)
        
        return outputs

    def summary(self):
        """打印模型结构摘要"""
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


"""
====================================================================================
【New_FM详解】
====================================================================================

New_FM是专门为DeepFM设计的FM层，与标准FM_Layer的区别在于：
- 它接收预计算好的Embedding（来自共享的Embedding层）
- 不需要自己维护隐向量V（由共享Embedding提供）

class New_FM(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        # feature_length: 所有特征值的总数量
        # w_reg: 一阶权重正则化
        
    def build(self, input_shape):
        # 只需要一阶权重w，二阶部分使用输入的Embedding
        self.w = self.add_weight(
            name='w', 
            shape=(self.feature_length, 1),  # 每个特征值一个权重
            ...
        )
    
    def call(self, inputs):
        # inputs包含:
        # - sparse_inputs: 特征的全局索引
        # - embed_inputs: 特征的Embedding向量
        
        # 一阶部分
        first_order = tf.reduce_sum(
            tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1
        )
        
        # 二阶部分（使用输入的Embedding）
        # square_sum = (Σ ei)^2
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1))
        
        # sum_square = Σ (ei)^2
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1)
        
        # 二阶 = 1/2 * (square_sum - sum_square)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=-1)
        
        return first_order + second_order

====================================================================================
【使用示例】
====================================================================================

from reclearn.data.feature_column import sparseFeature

# 1. 定义特征列（注意：所有embed_dim必须相同）
feature_columns = [
    sparseFeature('user_id', feat_num=10000, embed_dim=8),
    sparseFeature('item_id', feat_num=50000, embed_dim=8),
    sparseFeature('category', feat_num=100, embed_dim=8),
    sparseFeature('brand', feat_num=1000, embed_dim=8),
]

# 2. 准备数据
train_data = {
    'user_id': np.array([1, 2, 3, 4, 5]),
    'item_id': np.array([10, 20, 30, 40, 50]),
    'category': np.array([1, 2, 1, 3, 2]),
    'brand': np.array([5, 6, 7, 8, 9]),
}
labels = np.array([1, 0, 1, 0, 1])

# 3. 创建模型
model = DeepFM(
    feature_columns=feature_columns,
    hidden_units=(256, 128, 64),  # DNN隐藏层
    activation='relu',
    dnn_dropout=0.2,
    fm_w_reg=1e-6,
    embed_reg=1e-6
)

# 4. 编译和训练
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)
model.fit(x=train_data, y=labels, epochs=10, batch_size=256, validation_split=0.1)

# 5. 预测
predictions = model.predict(test_data)

====================================================================================
【DeepFM工程实践经验】
====================================================================================

1. Embedding维度选择
   - 通常设置为 8, 16, 32
   - 可以根据特征的取值个数调整（取值多的特征可以用更大的维度）
   - 但在标准DeepFM中，所有特征必须使用相同的维度

2. DNN层数和宽度
   - 通常2-4层足够
   - 每层256-512个单元
   - 可以使用递减结构，如(512, 256, 128)

3. 正则化
   - Dropout: 0.1-0.5
   - L2正则化: 1e-6 到 1e-4
   - 早停（Early Stopping）

4. 训练技巧
   - 使用Adam优化器
   - 学习率: 1e-3 到 1e-4
   - Batch Size: 256-2048

5. 特征工程
   - 虽然DeepFM可以自动特征交叉，但好的特征仍然重要
   - 可以添加统计特征（如点击率、转化率）
   - 可以添加用户/物品的历史统计

====================================================================================
【模型对比】
====================================================================================

┌─────────────────┬────────────────┬────────────────┬────────────────┐
│     模型         │     低阶交叉    │     高阶交叉    │   共享Embedding │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ LR              │ 手动           │ 无             │ -              │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ FM              │ 自动(二阶)     │ 无             │ -              │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ Wide&Deep       │ 手动           │ DNN           │ ✗              │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ DeepFM          │ 自动(FM)       │ DNN           │ ✓              │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ DCN             │ 交叉网络       │ DNN           │ ✓              │
├─────────────────┼────────────────┼────────────────┼────────────────┤
│ xDeepFM         │ CIN           │ DNN           │ ✓              │
└─────────────────┴────────────────┴────────────────┴────────────────┘

====================================================================================
"""

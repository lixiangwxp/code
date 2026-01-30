"""
====================================================================================
FM (Factorization Machines) - 因子分解机
====================================================================================
创建日期: August 25, 2020
更新日期: Nov, 11, 2021
参考论文: "Factorization Machines", ICDM, 2010
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

【模型简介】
FM是一种可以自动进行特征交叉的模型，是推荐系统中的经典算法。
它通过为每个特征学习一个低维向量，利用向量内积来表示特征交叉的权重。

【为什么需要特征交叉？】
在推荐和广告场景中，单个特征的预测能力有限，特征组合往往更有效：
- 例如："男性" + "足球" -> 点击率高
- 例如："女性" + "口红" -> 点击率高

传统方法（多项式回归）的问题：
- 需要手动设计交叉特征
- 稀疏数据下交叉特征难以学习（很多特征组合没有出现过）

【FM的核心思想】
不直接学习每对特征的交互权重w_ij，而是学习每个特征的隐向量v_i，
然后用内积<v_i, v_j>来近似交互权重。

【数学公式】
线性回归：y = w0 + Σ wi*xi
多项式回归：y = w0 + Σ wi*xi + Σ Σ wij*xi*xj    (需要学习 n^2 个参数)
FM：y = w0 + Σ wi*xi + Σ Σ <vi, vj>*xi*xj       (只需要学习 n*k 个参数)

其中：
- w0: 全局偏置
- wi: 第i个特征的一阶权重
- vi: 第i个特征的隐向量 (维度为k)
- <vi, vj>: 向量vi和vj的内积

【FM公式的高效计算】
原始公式的时间复杂度是O(n^2*k)，但可以简化为O(n*k)：

ΣΣ <vi, vj>*xi*xj = 1/2 * Σ[(Σ vi*xi)^2 - Σ (vi*xi)^2]
                      f     i             i

【模型结构图】

    输入: 特征向量 [x1, x2, x3, ..., xn] (通常是one-hot编码)
                            ↓
    ┌────────────────────────────────────────────┐
    │                一阶部分 (Linear)             │
    │        w0 + Σ wi*xi                         │
    │        直接学习每个特征的权重                 │
    └────────────────────────────────────────────┘
                            +
    ┌────────────────────────────────────────────┐
    │              二阶交叉部分 (FM)               │
    │      1/2 * Σ[(Σ vi*xi)^2 - Σ(vi*xi)^2]     │
    │      通过隐向量内积表示特征交叉               │
    └────────────────────────────────────────────┘
                            ↓
                     sigmoid(output)
                            ↓
                      点击概率 [0, 1]

【输入数据格式】
{
    'feat1': [val1, val2, ...],  # 特征1的值（通常是类别ID）
    'feat2': [val1, val2, ...],  # 特征2的值
    ...
}
====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import FM_Layer


class FM(Model):
    """
    FM因子分解机模型
    
    FM的核心优势：
    1. 自动进行特征交叉，无需手动特征工程
    2. 在稀疏数据上表现良好（通过隐向量泛化）
    3. 计算效率高，线性时间复杂度
    """
    
    def __init__(self, feature_columns, k=8, w_reg=0., v_reg=0.):
        """
        初始化FM模型
        
        参数说明:
        ---------
        feature_columns : list
            特征列配置列表，每个元素是一个字典：
            [
                {'feat_name': 'user_id', 'feat_num': 10000, 'embed_dim': 8},
                {'feat_name': 'item_id', 'feat_num': 50000, 'embed_dim': 8},
                ...
            ]
            - feat_name: 特征名称
            - feat_num: 该特征的取值个数（如用户ID的最大值+1）
            - embed_dim: 嵌入维度（这里不直接使用，但保持格式一致）
            
        k : int, 默认8
            隐向量的维度
            - 维度越大，模型表达能力越强，但参数也更多
            - 常用值: 4, 8, 16, 32
            
        w_reg : float, 默认0.0
            一阶权重w的L2正则化系数
            用于防止过拟合
            
        v_reg : float, 默认0.0
            隐向量V的L2正则化系数
            用于防止过拟合
        """
        super(FM, self).__init__()
        
        # 保存特征列配置
        self.feature_columns = feature_columns
        
        # FM核心层：包含一阶线性部分和二阶交叉部分
        self.fm = FM_Layer(
            feature_columns=feature_columns,
            k=k,              # 隐向量维度
            w_reg=w_reg,      # 一阶权重正则化
            v_reg=v_reg       # 隐向量正则化
        )

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
                'hour': [8, 9, 10, ...]         # shape: (batch_size,)
            }
        
        返回:
        -----
        outputs : Tensor
            预测的点击概率，shape: (batch_size, 1)
            值域: [0, 1]
        
        处理流程:
        ---------
        1. FM层计算一阶和二阶得分
        2. Sigmoid激活得到概率
        """
        
        # ===== Step 1: FM层计算 =====
        # FM层内部会：
        # 1) 计算一阶部分: w0 + Σ wi*xi
        # 2) 计算二阶交叉部分: 1/2 * Σ[(Σ vi*xi)^2 - Σ(vi*xi)^2]
        # 3) 返回两者之和
        fm_outputs = self.fm(inputs)  # shape: (batch_size, 1)
        
        # ===== Step 2: Sigmoid激活 =====
        # 将得分转换为概率
        outputs = tf.nn.sigmoid(fm_outputs)  # shape: (batch_size, 1)
        
        return outputs

    def summary(self):
        """打印模型结构摘要"""
        # 为每个特征创建输入占位符
        inputs = {
            feat['feat_name']: Input(shape=(), dtype=tf.int32, name=feat['feat_name'])
            for feat in self.feature_columns
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


"""
====================================================================================
【FM_Layer详解】
====================================================================================

FM_Layer是FM的核心实现，让我们深入了解其内部工作原理：

class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        '''
        参数:
        - feature_columns: 特征配置
        - k: 隐向量维度
        - w_reg: 一阶权重正则化
        - v_reg: 隐向量正则化
        '''
        # 计算所有特征的总数量（用于初始化参数）
        self.feature_length = sum(feat['feat_num'] for feat in feature_columns)
        
    def build(self, input_shape):
        # w0: 全局偏置，shape: (1,)
        self.w0 = self.add_weight(name='w0', shape=(1,))
        
        # w: 一阶权重，shape: (feature_length, 1)
        # 每个特征值都有一个权重
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1))
        
        # V: 隐向量矩阵，shape: (feature_length, k)
        # 每个特征值都有一个k维的隐向量
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k))
    
    def call(self, inputs):
        # 1. 计算一阶部分
        # embedding_lookup获取每个特征的一阶权重
        first_order = self.w0 + tf.reduce_sum(
            tf.nn.embedding_lookup(self.w, inputs), axis=1
        )  # (batch_size, 1)
        
        # 2. 计算二阶部分
        # 获取每个特征的隐向量
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, k)
        
        # 利用公式: ΣΣ<vi,vj>*xi*xj = 1/2 * [(Σvi*xi)^2 - Σ(vi*xi)^2]
        # 因为xi都是1（one-hot），所以简化为:
        # square_sum = (Σvi)^2
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1))  # (batch_size, k)
        
        # sum_square = Σ(vi)^2
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1)  # (batch_size, k)
        
        # 二阶部分 = 1/2 * (square_sum - sum_square)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1)  # (batch_size, 1)
        
        return first_order + second_order

====================================================================================
【FM公式推导】
====================================================================================

原始二阶交叉公式（复杂度O(n^2*k)）:
    Σ_i Σ_j>i <v_i, v_j> * x_i * x_j

等价变换（复杂度O(n*k)）:
    = 1/2 * Σ_i Σ_j [<v_i, v_j> * x_i * x_j - Σ_i <v_i, v_i> * x_i^2]
    = 1/2 * [Σ_i Σ_j <v_i, v_j> * x_i * x_j - Σ_i <v_i, v_i> * x_i^2]
    = 1/2 * [Σ_f (Σ_i v_if * x_i)^2 - Σ_i (v_if * x_i)^2]

其中f是隐向量的第f个维度，i是第i个特征。

====================================================================================
【使用示例】
====================================================================================

from reclearn.data.feature_column import sparseFeature

# 1. 定义特征列
feature_columns = [
    sparseFeature('user_id', feat_num=10000, embed_dim=8),
    sparseFeature('item_id', feat_num=50000, embed_dim=8),
    sparseFeature('category', feat_num=100, embed_dim=8),
    sparseFeature('hour', feat_num=24, embed_dim=8),
]

# 2. 准备数据
train_data = {
    'user_id': np.array([1, 2, 3, 4, 5]),
    'item_id': np.array([10, 20, 30, 40, 50]),
    'category': np.array([1, 2, 1, 3, 2]),
    'hour': np.array([8, 9, 10, 11, 12]),
}
labels = np.array([1, 0, 1, 0, 1])  # 是否点击

# 3. 创建模型
model = FM(
    feature_columns=feature_columns,
    k=8,           # 隐向量维度
    w_reg=1e-6,    # 一阶正则化
    v_reg=1e-6     # 二阶正则化
)

# 4. 编译和训练
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC']
)
model.fit(x=train_data, y=labels, epochs=10, batch_size=256)

# 5. 预测
predictions = model.predict(test_data)

====================================================================================
【FM vs 其他模型对比】
====================================================================================

┌──────────────┬────────────────┬─────────────────┬─────────────────────┐
│    模型       │    特征交叉     │    参数量        │    优缺点            │
├──────────────┼────────────────┼─────────────────┼─────────────────────┤
│ 逻辑回归 LR   │ 无(需手动)     │ O(n)            │ 简单，但无法自动交叉  │
├──────────────┼────────────────┼─────────────────┼─────────────────────┤
│ Poly2       │ 显式交叉       │ O(n^2)          │ 稀疏数据效果差       │
├──────────────┼────────────────┼─────────────────┼─────────────────────┤
│ FM          │ 隐式交叉       │ O(n*k)          │ 泛化能力强          │
├──────────────┼────────────────┼─────────────────┼─────────────────────┤
│ FFM         │ 场感知交叉     │ O(n*f*k)        │ 更精细但更复杂       │
├──────────────┼────────────────┼─────────────────┼─────────────────────┤
│ DeepFM      │ FM + DNN      │ O(n*k) + DNN    │ 兼顾记忆和泛化       │
└──────────────┴────────────────┴─────────────────┴─────────────────────┘

其中：
- n: 特征数量
- k: 隐向量维度
- f: 特征域（field）数量

====================================================================================
"""

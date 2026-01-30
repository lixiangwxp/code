"""
====================================================================================
SASRec (Self-Attentive Sequential Recommendation) - 自注意力序列推荐模型
====================================================================================
创建日期: Dec 20, 2020
更新日期: Apr 22, 2022
参考论文: "Self-Attentive Sequential Recommendation", ICDM, 2018
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

【模型简介】
SASRec是一种基于自注意力机制（Self-Attention）的序列推荐模型。
它使用Transformer的编码器结构来建模用户的历史行为序列，
能够捕捉序列中物品之间的长距离依赖关系。

【与RNN方法的对比】
┌─────────────────────┬────────────────────────────────────────┐
│     GRU4Rec (RNN)   │              SASRec (Transformer)      │
├─────────────────────┼────────────────────────────────────────┤
│ 顺序处理，速度慢      │ 并行处理，速度快                         │
│ 难以捕捉长距离依赖    │ 自注意力可直接建模任意距离的依赖           │
│ 隐状态会逐渐遗忘      │ 所有位置都可以直接访问                    │
│ 相对简单              │ 需要位置编码                             │
└─────────────────────┴────────────────────────────────────────┘

【模型结构图】

    输入: 用户历史点击序列 [item_1, item_2, item_3, ..., item_n]
                                    ↓
    ┌─────────────────────────────────────────────────────────┐
    │                    Embedding Layer                       │
    │   item_embed + position_embed                            │
    └─────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────┐
    │                  Transformer Encoder                     │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │            Multi-Head Self-Attention             │   │
    │   │   Q = K = V = 序列嵌入                            │   │
    │   └─────────────────────────────────────────────────┘   │
    │                          ↓                               │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │           Add & Layer Normalization              │   │
    │   └─────────────────────────────────────────────────┘   │
    │                          ↓                               │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │              Feed Forward Network                │   │
    │   └─────────────────────────────────────────────────┘   │
    │                          ↓                               │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │           Add & Layer Normalization              │   │
    │   └─────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘
                                    ↓
                            (重复 blocks 次)
                                    ↓
    ┌─────────────────────────────────────────────────────────┐
    │        取最后一个位置的输出作为用户向量                     │
    │        user_vector = outputs[:, -1, :]                   │
    └─────────────────────────────────────────────────────────┘
                                    ↓
                        与候选物品计算内积得分
                                    ↓
                                 输出

【输入数据格式】
{
    'click_seq': [[item1, item2, ...], ...],  # 点击序列, shape: (batch_size, seq_len)
    'pos_item': [pos_item_1, ...],            # 正样本物品ID
    'neg_item': [[neg1, neg2], ...]           # 负样本物品ID
}
====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.layers import TransformerEncoder
from reclearn.models.losses import get_loss


class SASRec(Model):
    """
    SASRec自注意力序列推荐模型
    
    核心创新：
    1. 使用Transformer Encoder替代RNN处理序列
    2. 引入位置编码捕获序列顺序信息
    3. 使用多头自注意力捕获物品间的关系
    """
    
    def __init__(self, item_num, embed_dim, seq_len=100, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dnn_dropout=0., layer_norm_eps=1e-6, use_l2norm=False,
                 loss_name="binary_cross_entropy_loss", gamma=0.5, embed_reg=0., seed=None):
        """
        初始化SASRec模型
        
        参数说明:
        ---------
        item_num : int
            物品数量（最大物品ID + 1）
            
        embed_dim : int
            嵌入向量维度
            这同时也是Transformer的d_model参数
            
        seq_len : int, 默认100
            输入序列的固定长度
            短于此长度的序列会用0填充（padding）
            长于此长度的序列会截断
            
        blocks : int, 默认1
            Transformer Encoder的层数
            层数越多，模型越深，表达能力越强
            常用值: 1, 2, 3
            
        num_heads : int, 默认1
            多头注意力的头数
            要求: embed_dim 能被 num_heads 整除
            常用值: 1, 2, 4, 8
            
        ffn_hidden_unit : int, 默认128
            FFN（前馈神经网络）的隐藏层维度
            通常设置为 embed_dim 的2-4倍
            
        dnn_dropout : float, 默认0.0
            Dropout比例，用于正则化
            
        layer_norm_eps : float, 默认1e-6
            Layer Normalization的epsilon参数
            防止除零错误
            
        use_l2norm : bool, 默认False
            是否对输出向量进行L2归一化
            
        loss_name : str, 默认"binary_cross_entropy_loss"
            损失函数名称
            
        gamma : float, 默认0.5
            hinge_loss的margin参数
            
        embed_reg : float, 默认0.0
            嵌入层L2正则化系数
            
        seed : int, 可选
            随机种子
        """
        super(SASRec, self).__init__()
        
        # ===== 物品嵌入层 =====
        # 将物品ID映射为稠密向量
        self.item_embedding = Embedding(
            input_dim=item_num,              # 物品总数
            input_length=1,
            output_dim=embed_dim,            # 嵌入维度
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        
        # ===== 位置嵌入层 =====
        # 为序列中的每个位置学习一个位置向量
        # 这是可学习的位置编码，与原始Transformer的固定位置编码不同
        self.pos_embedding = Embedding(
            input_dim=seq_len,               # 最大序列长度
            input_length=1,
            output_dim=embed_dim,            # 与物品嵌入维度相同
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        
        # Dropout层
        self.dropout = Dropout(dnn_dropout)
        
        # ===== Transformer Encoder层 =====
        # 创建多个Encoder层（根据blocks参数）
        self.encoder_layer = [
            TransformerEncoder(
                d_model=embed_dim,           # 模型维度
                num_heads=num_heads,         # 注意力头数
                ffn_hidden_unit=ffn_hidden_unit,  # FFN隐藏层维度
                dropout=dnn_dropout,         # Dropout比例
                layer_norm_eps=layer_norm_eps     # LayerNorm epsilon
            ) 
            for _ in range(blocks)           # 创建blocks个Encoder层
        ]
        
        # 保存配置
        self.use_l2norm = use_l2norm
        self.loss_name = loss_name
        self.gamma = gamma
        self.seq_len = seq_len
        
        # 设置随机种子
        tf.random.set_seed(seed)

    def call(self, inputs):
        """
        模型前向传播
        
        参数:
        -----
        inputs : dict
            输入字典，包含:
            - 'click_seq': 用户点击序列，shape=(batch_size, seq_len)
            - 'pos_item': 正样本物品ID，shape=(batch_size,)
            - 'neg_item': 负样本物品ID，shape=(batch_size, neg_num)
        
        返回:
        -----
        logits : Tensor
            预测得分，shape=(batch_size, 1+neg_num)
        
        处理流程:
        ---------
        1. 序列嵌入 = 物品嵌入 + 位置嵌入
        2. 创建mask（标记padding位置）
        3. 通过多层Transformer Encoder
        4. 取最后位置的输出作为用户向量
        5. 与候选物品计算得分
        6. 计算损失
        """
        
        # ==================== Step 1: 序列嵌入 ====================
        
        # 获取序列中每个物品的嵌入向量
        # 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, embed_dim)
        seq_embed = self.item_embedding(inputs['click_seq'])
        
        # ==================== Step 2: 创建Mask ====================
        
        # Mask用于标记哪些位置是padding（物品ID为0表示padding）
        # tf.not_equal(inputs['click_seq'], 0): 非0位置为True
        # 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, 1)
        mask = tf.expand_dims(
            tf.cast(tf.not_equal(inputs['click_seq'], 0), dtype=tf.float32), 
            axis=-1
        )
        
        # ==================== Step 3: 添加位置编码 ====================
        
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        # tf.range(self.seq_len): 生成位置索引
        # pos_embedding: 查表得到位置向量
        # tf.expand_dims(..., axis=0): 增加batch维度以便广播
        pos_encoding = tf.expand_dims(
            self.pos_embedding(tf.range(self.seq_len)), 
            axis=0
        )  # shape: (1, seq_len, embed_dim)
        
        # 物品嵌入 + 位置嵌入（广播加法）
        # seq_embed: (batch_size, seq_len, embed_dim)
        # pos_encoding: (1, seq_len, embed_dim) -> 广播到 (batch_size, seq_len, embed_dim)
        seq_embed += pos_encoding
        
        # Dropout
        seq_embed = self.dropout(seq_embed)
        
        # 初始化attention输出
        att_outputs = seq_embed  # shape: (batch_size, seq_len, embed_dim)
        
        # 应用mask（将padding位置的值置为0）
        att_outputs *= mask
        
        # ==================== Step 4: Transformer Encoder ====================
        
        # 通过多层Transformer Encoder
        for block in self.encoder_layer:
            # 每个block接收 [输入, mask]
            att_outputs = block([att_outputs, mask])  # (batch_size, seq_len, embed_dim)
            # 再次应用mask
            att_outputs *= mask
        
        # ==================== Step 5: 获取用户向量 ====================
        
        # 方法1: 取序列最后一个位置的输出作为用户向量
        # 这个向量编码了用户的整体兴趣
        # tf.slice: 从att_outputs中切片
        # begin=[0, seq_len-1, 0]: 从最后一个位置开始
        # size=[-1, 1, -1]: 取所有batch、1个位置、所有维度
        user_info = tf.slice(
            att_outputs, 
            begin=[0, self.seq_len-1, 0], 
            size=[-1, 1, -1]
        )  # shape: (batch_size, 1, embed_dim)
        
        # 方法2（注释掉的）: 取所有位置的平均值
        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (batch_size, embed_dim)
        
        # ==================== Step 6: 获取候选物品向量 ====================
        
        # 正样本物品嵌入
        # 输入: (batch_size,) -> 输出: (batch_size, embed_dim)
        pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))
        
        # 负样本物品嵌入
        # 输入: (batch_size, neg_num) -> 输出: (batch_size, neg_num, embed_dim)
        neg_info = self.item_embedding(inputs['neg_item'])
        
        # ==================== Step 7: L2归一化（可选） ====================
        
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_info = tf.math.l2_normalize(user_info, axis=-1)
        
        # ==================== Step 8: 计算得分 ====================
        
        # 正样本得分
        # user_info: (batch_size, 1, embed_dim)
        # pos_info: (batch_size, embed_dim) -> 扩展为 (batch_size, 1, embed_dim)
        pos_scores = tf.reduce_sum(
            tf.multiply(user_info, tf.expand_dims(pos_info, axis=1)),
            axis=-1
        )  # shape: (batch_size, 1)
        
        # 负样本得分
        # user_info: (batch_size, 1, embed_dim)
        # neg_info: (batch_size, neg_num, embed_dim)
        neg_scores = tf.reduce_sum(
            tf.multiply(user_info, neg_info),
            axis=-1
        )  # shape: (batch_size, neg_num)
        
        # ==================== Step 9: 计算损失 ====================
        
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        
        # 拼接输出
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        
        return logits

    def summary(self):
        """打印模型结构摘要"""
        inputs = {
            'click_seq': Input(shape=(self.seq_len,), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


"""
====================================================================================
【自注意力机制详解】
====================================================================================

自注意力的核心公式:

    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

其中:
- Q (Query): 查询向量，用于查询信息
- K (Key): 键向量，用于被查询匹配
- V (Value): 值向量，实际的信息内容
- d_k: K的维度，用于缩放

在SASRec中:
- Q = K = V = 序列嵌入（自注意力的特点）
- 每个位置的输出是序列中所有位置的加权和
- 权重由该位置与其他位置的相关性决定

┌───────────────────────────────────────────────────────────────┐
│                    自注意力可视化                               │
│                                                                │
│  输入序列: [手机壳, 手机膜, 耳机, 充电器, 笔记本]                │
│                                                                │
│  注意力权重矩阵:                                                │
│                手机壳  手机膜  耳机  充电器  笔记本              │
│  手机壳        0.3    0.3    0.1   0.2    0.1                 │
│  手机膜        0.3    0.3    0.1   0.2    0.1                 │
│  耳机          0.2    0.2    0.3   0.2    0.1                 │
│  充电器        0.2    0.2    0.1   0.3    0.2                 │
│  笔记本        0.1    0.1    0.1   0.2    0.5                 │
│                                                                │
│  观察: 相似物品（如手机壳和手机膜）之间的注意力权重较高          │
└───────────────────────────────────────────────────────────────┘

====================================================================================
【位置编码的作用】
====================================================================================

为什么需要位置编码?
- 自注意力本身不考虑位置信息（是排列不变的）
- 但在推荐中，用户行为的顺序很重要
- 位置编码让模型知道每个物品在序列中的位置

两种位置编码方式:
1. 固定位置编码（原始Transformer）:
   PE(pos, 2i) = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

2. 可学习位置编码（SASRec使用）:
   直接为每个位置学习一个向量

====================================================================================
【使用示例】
====================================================================================

# 1. 准备数据
train_data = {
    'click_seq': np.array([
        [1, 2, 3, 0, 0],   # 用户1点击了物品1,2,3
        [4, 5, 0, 0, 0],   # 用户2点击了物品4,5
    ]),
    'pos_item': np.array([4, 6]),      # 下一个点击的物品
    'neg_item': np.array([[5, 6], [7, 8]])  # 负样本
}

# 2. 创建模型
model = SASRec(
    item_num=10000,        # 物品数量
    embed_dim=64,          # 嵌入维度
    seq_len=100,           # 序列长度
    blocks=2,              # 2层Encoder
    num_heads=4,           # 4个注意力头
    ffn_hidden_unit=256,   # FFN隐藏层
    dnn_dropout=0.2,       # Dropout
    use_l2norm=True        # L2归一化
)

# 3. 训练
model.compile(optimizer='adam')
model.fit(x=train_data, epochs=20, batch_size=256)

====================================================================================
"""

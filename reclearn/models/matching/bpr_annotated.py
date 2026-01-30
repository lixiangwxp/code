"""
====================================================================================
BPR (Bayesian Personalized Ranking) - 贝叶斯个性化排序模型
====================================================================================
创建日期: Nov 13, 2020
更新日期: Apr 9, 2022
参考论文: "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI, 2009
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

【模型简介】
BPR是一种基于矩阵分解的推荐算法，专门用于隐式反馈数据（如点击、购买等）。
核心思想：对于每个用户，让其交互过的物品得分高于未交互的物品。

【模型结构】
    用户嵌入表                物品嵌入表
        ↓                        ↓
    user_embed              item_embed (正样本/负样本)
        ↓                        ↓
        └────────→ 内积 ←────────┘
                    ↓
              得分(scores)
                    ↓
              BPR损失函数

【输入数据格式】
{
    'user': [user_id_1, user_id_2, ...],           # 用户ID，shape: (batch_size,)
    'pos_item': [pos_item_1, pos_item_2, ...],     # 正样本物品ID，shape: (batch_size,)
    'neg_item': [[neg1, neg2], [neg1, neg2], ...]  # 负样本物品ID，shape: (batch_size, neg_num)
}

【输出】
logits: 正样本和负样本的得分，shape: (batch_size, 1 + neg_num)
        第一列是正样本得分，后面是负样本得分
====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2

from reclearn.models.losses import bpr_loss


class BPR(Model):
    """
    BPR模型类 - 继承自Keras Model
    
    这是一个基于矩阵分解的协同过滤模型，通过学习用户和物品的低维向量表示，
    使得用户喜欢的物品得分高于不喜欢的物品。
    """
    
    def __init__(self, user_num, item_num, embed_dim, use_l2norm=False, embed_reg=0., seed=None):
        """
        初始化BPR模型
        
        参数说明:
        ---------
        user_num : int
            用户数量。注意：应该是最大用户ID + 1，因为ID从0开始
            例如：用户ID范围是1-6040，则user_num=6041
            
        item_num : int
            物品数量。同样是最大物品ID + 1
            
        embed_dim : int
            嵌入向量维度。常用值：32, 64, 128
            维度越大，模型表达能力越强，但也更容易过拟合
            
        use_l2norm : bool, 默认False
            是否对嵌入向量进行L2归一化
            归一化后向量模长为1，可以使得相似度计算更稳定
            
        embed_reg : float, 默认0.0
            嵌入层的L2正则化系数
            用于防止过拟合，常用值：1e-6, 1e-5
            
        seed : int, 可选
            随机种子，用于结果复现
        """
        # 调用父类初始化方法
        super(BPR, self).__init__()
        
        # ===== 用户嵌入层 =====
        # 将用户ID映射为稠密向量
        # input_dim: 词表大小（用户数量）
        # input_length: 输入序列长度（这里是1，因为每次只输入一个用户ID）
        # output_dim: 输出向量维度
        # embeddings_initializer: 初始化方式
        # embeddings_regularizer: 正则化方式
        self.user_embedding = Embedding(
            input_dim=user_num,           # 用户数量
            input_length=1,               # 每次输入1个用户ID
            output_dim=embed_dim,         # 嵌入维度
            embeddings_initializer='random_normal',  # 随机正态分布初始化
            embeddings_regularizer=l2(embed_reg)     # L2正则化
        )
        
        # ===== 物品嵌入层 =====
        # 将物品ID映射为稠密向量
        # 与用户嵌入维度相同，以便计算内积
        self.item_embedding = Embedding(
            input_dim=item_num,           # 物品数量
            input_length=1,               # 每次输入1个物品ID
            output_dim=embed_dim,         # 嵌入维度（与用户相同）
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        
        # 是否使用L2归一化
        self.use_l2norm = use_l2norm
        
        # 设置随机种子（用于结果复现）
        tf.random.set_seed(seed)

    def call(self, inputs):
        """
        模型前向传播
        
        参数:
        -----
        inputs : dict
            输入字典，包含:
            - 'user': 用户ID张量，shape=(batch_size,)
            - 'pos_item': 正样本物品ID张量，shape=(batch_size,)
            - 'neg_item': 负样本物品ID张量，shape=(batch_size, neg_num)
        
        返回:
        -----
        logits : Tensor
            预测得分，shape=(batch_size, 1+neg_num)
            第0列是正样本得分，其余列是负样本得分
        
        处理流程:
        ---------
        1. 获取用户嵌入向量
        2. 获取正样本物品嵌入向量
        3. 获取负样本物品嵌入向量
        4. (可选) L2归一化
        5. 计算用户与正/负样本的内积得分
        6. 计算BPR损失并添加到模型
        7. 返回得分
        """
        
        # ===== Step 1: 获取用户嵌入向量 =====
        # tf.reshape(..., [-1, ]): 将输入展平为1维向量
        # 输入shape: (batch_size,) -> 输出shape: (batch_size, embed_dim)
        user_embed = self.user_embedding(tf.reshape(inputs['user'], [-1, ]))
        
        # ===== Step 2: 获取正样本物品嵌入向量 =====
        # 正样本是用户实际交互过的物品
        # 输入shape: (batch_size,) -> 输出shape: (batch_size, embed_dim)
        pos_info = self.item_embedding(tf.reshape(inputs['pos_item'], [-1, ]))
        
        # ===== Step 3: 获取负样本物品嵌入向量 =====
        # 负样本是随机采样的用户未交互物品
        # 输入shape: (batch_size, neg_num) -> 输出shape: (batch_size, neg_num, embed_dim)
        neg_info = self.item_embedding(inputs['neg_item'])
        
        # ===== Step 4: L2归一化（可选） =====
        # 归一化后，向量的模长为1，内积变成了余弦相似度
        if self.use_l2norm:
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)   # 在最后一个维度归一化
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
            user_embed = tf.math.l2_normalize(user_embed, axis=-1)
        
        # ===== Step 5: 计算得分 =====
        # 正样本得分: 用户向量与正样本向量的内积
        # tf.multiply: 逐元素相乘
        # tf.reduce_sum(..., axis=-1): 在最后一维求和，得到内积
        # keepdims=True: 保持维度，结果shape=(batch_size, 1)
        pos_scores = tf.reduce_sum(
            tf.multiply(user_embed, pos_info),  # (batch_size, embed_dim)
            axis=-1, 
            keepdims=True
        )  # 输出shape: (batch_size, 1)
        
        # 负样本得分: 用户向量与每个负样本向量的内积
        # tf.expand_dims(user_embed, axis=1): 增加一维以便广播
        # user_embed shape: (batch_size, embed_dim) -> (batch_size, 1, embed_dim)
        # neg_info shape: (batch_size, neg_num, embed_dim)
        # 广播乘法后: (batch_size, neg_num, embed_dim)
        # reduce_sum后: (batch_size, neg_num)
        neg_scores = tf.reduce_sum(
            tf.multiply(tf.expand_dims(user_embed, axis=1), neg_info),
            axis=-1
        )  # 输出shape: (batch_size, neg_num)
        
        # ===== Step 6: 计算并添加BPR损失 =====
        # BPR损失公式: -log(sigmoid(pos_score - neg_score))
        # 含义: 让正样本得分尽可能高于负样本得分
        self.add_loss(bpr_loss(pos_scores, neg_scores))
        
        # ===== Step 7: 拼接并返回得分 =====
        # 将正样本得分和负样本得分拼接
        # 用于评估时计算排名
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        # 输出shape: (batch_size, 1 + neg_num)
        
        return logits

    def get_user_vector(self, inputs):
        """
        获取用户向量（用于在线服务时提取用户表示）
        
        参数:
        -----
        inputs : dict
            包含'user'键的字典
            
        返回:
        -----
        user_vector : Tensor
            用户嵌入向量，shape=(batch_size, embed_dim)
        """
        if len(inputs) < 2 and inputs.get('user') is not None:
            return self.user_embedding(inputs['user'])

    def summary(self):
        """
        打印模型结构摘要
        
        由于Keras Model需要知道输入shape才能打印summary，
        这里我们构建一个临时的函数式模型来显示结构
        """
        # 定义输入占位符
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),          # 用户ID
            'pos_item': Input(shape=(), dtype=tf.int32),      # 正样本物品ID
            'neg_item': Input(shape=(1,), dtype=tf.int32)     # 负样本物品ID (假设neg_num=1)
        }
        # 构建临时模型并打印摘要
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


"""
====================================================================================
使用示例
====================================================================================

# 1. 导入必要模块
from reclearn.models.matching import BPR
from reclearn.data.datasets import movielens as ml

# 2. 准备数据
train_data = {
    'user': np.array([1, 2, 3, ...]),           # 用户ID
    'pos_item': np.array([10, 20, 30, ...]),    # 正样本物品ID
    'neg_item': np.array([[5, 6], [15, 16], [25, 26], ...])  # 负样本物品ID
}

# 3. 创建模型
model = BPR(
    user_num=6041,      # 用户数量
    item_num=3953,      # 物品数量
    embed_dim=64,       # 嵌入维度
    use_l2norm=False,   # 是否L2归一化
    embed_reg=1e-6      # 正则化系数
)

# 4. 编译模型
model.compile(optimizer='adam')

# 5. 训练模型
model.fit(x=train_data, epochs=20, batch_size=512)

# 6. 评估模型
from reclearn.evaluator import eval_pos_neg
result = eval_pos_neg(model, test_data, ['hr', 'ndcg'], k=10)
print(f"HR@10: {result['hr']:.4f}, NDCG@10: {result['ndcg']:.4f}")

====================================================================================
"""

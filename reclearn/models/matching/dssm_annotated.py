"""
====================================================================================
DSSM (Deep Structured Semantic Model) - 深度结构化语义模型（双塔模型）
====================================================================================
创建日期: Mar 31, 2022
参考论文: "Learning Deep Structured Semantic Models for Web Search using Clickthrough Data", CIKM, 2013
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

【模型简介】
DSSM是工业界常用的召回模型，也被称为"双塔模型"。
它将用户和物品分别通过独立的深度神经网络编码成向量，
然后通过计算向量的相似度（如内积或余弦相似度）来衡量匹配程度。

【为什么叫双塔？】
因为模型结构像两座塔：
- 左塔（User Tower）：负责编码用户特征
- 右塔（Item Tower）：负责编码物品特征
两塔独立，可以离线预计算，非常适合工业应用。

【模型结构图】

    User Tower                          Item Tower
    ─────────                          ─────────
    
    user_id                            item_id
       ↓                                  ↓
  ┌─────────┐                       ┌─────────┐
  │Embedding│                       │Embedding│
  └────┬────┘                       └────┬────┘
       ↓                                  ↓
  ┌─────────┐                       ┌─────────┐
  │  MLP    │                       │  MLP    │
  │ Layer1  │                       │ Layer1  │
  └────┬────┘                       └────┬────┘
       ↓                                  ↓
  ┌─────────┐                       ┌─────────┐
  │  MLP    │                       │  MLP    │
  │ Layer2  │                       │ Layer2  │
  └────┬────┘                       └────┬────┘
       ↓                                  ↓
  user_vector                       item_vector
       │                                  │
       └──────────→ 内积 ←────────────────┘
                     ↓
                   score

【输入数据格式】
{
    'user': [user_id_1, user_id_2, ...],           # 用户ID
    'pos_item': [pos_item_1, pos_item_2, ...],     # 正样本物品ID
    'neg_item': [[neg1, neg2], [neg1, neg2], ...]  # 负样本物品ID
}

【工业应用场景】
1. 用户塔可以实时计算用户向量
2. 物品塔可以离线计算所有物品向量并建立索引（如Faiss）
3. 线上通过向量检索快速召回Top-K候选物品
====================================================================================
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.regularizers import l2
from reclearn.layers import MLP
from reclearn.models.losses import get_loss


class DSSM(Model):
    """
    DSSM双塔模型类
    
    双塔模型是工业界最常用的召回模型之一，其核心优势是：
    1. 用户塔和物品塔相互独立，可以分别预计算
    2. 物品向量可以离线计算并建立向量索引
    3. 在线只需要计算用户向量，然后通过向量检索召回候选集
    """
    
    def __init__(self, user_num, item_num, embed_dim, user_mlp, item_mlp, activation='relu',
                 dnn_dropout=0., use_l2norm=False, loss_name="binary_cross_entropy_loss",
                 gamma=0.5, embed_reg=0., seed=None):
        """
        初始化DSSM模型
        
        参数说明:
        ---------
        user_num : int
            用户数量（最大用户ID + 1）
            
        item_num : int
            物品数量（最大物品ID + 1）
            
        embed_dim : int
            嵌入向量维度
            这是用户ID和物品ID经过Embedding后的初始维度
            
        user_mlp : list
            用户塔MLP的隐藏层单元数列表
            例如: [128, 64, 32] 表示3层MLP，单元数分别为128、64、32
            ⚠️ 注意: 最后一层的维度必须与item_mlp最后一层相同！
            
        item_mlp : list
            物品塔MLP的隐藏层单元数列表
            最后一层维度必须与user_mlp相同，因为需要计算向量内积
            
        activation : str, 默认'relu'
            MLP激活函数名称
            可选: 'relu', 'sigmoid', 'tanh', 'leaky_relu'等
            
        dnn_dropout : float, 默认0.0
            MLP的Dropout比例（0-1之间）
            用于防止过拟合
            
        use_l2norm : bool, 默认False
            是否对最终向量进行L2归一化
            归一化后内积等于余弦相似度，取值范围[-1, 1]
            
        loss_name : str, 默认"binary_cross_entropy_loss"
            损失函数名称，可选:
            - "binary_cross_entropy_loss": 二元交叉熵（点对点）
            - "bpr_loss": BPR损失（成对）
            - "hinge_loss": 合页损失（成对）
            
        gamma : float, 默认0.5
            hinge_loss的margin参数
            只在loss_name="hinge_loss"时有效
            
        embed_reg : float, 默认0.0
            嵌入层L2正则化系数
            
        seed : int, 可选
            随机种子
        """
        super(DSSM, self).__init__()
        
        # ===== 参数检查 =====
        # 确保用户塔和物品塔输出维度相同
        if user_mlp[-1] != item_mlp[-1]:
            raise ValueError("The last value of user_mlp must be equal to item_mlp's.")
        
        # ===== 用户嵌入表 =====
        # 将用户ID映射为稠密向量
        self.user_embedding_table = Embedding(
            input_dim=user_num,              # 用户总数
            input_length=1,                  # 输入长度
            output_dim=embed_dim,            # 嵌入维度
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        
        # ===== 物品嵌入表 =====
        # 将物品ID映射为稠密向量
        self.item_embedding_table = Embedding(
            input_dim=item_num,              # 物品总数
            input_length=1,                  # 输入长度
            output_dim=embed_dim,            # 嵌入维度
            embeddings_initializer='random_normal',
            embeddings_regularizer=l2(embed_reg)
        )
        
        # ===== 用户塔MLP =====
        # 对用户嵌入进行非线性变换，学习更高层次的用户表示
        self.user_mlp_layer = MLP(
            hidden_units=user_mlp,           # 如[128, 64, 32]
            activation=activation,           # 激活函数
            dnn_dropout=dnn_dropout          # Dropout比例
        )
        
        # ===== 物品塔MLP =====
        # 对物品嵌入进行非线性变换
        self.item_mlp_layer = MLP(
            hidden_units=item_mlp,           # 如[128, 64, 32]
            activation=activation,
            dnn_dropout=dnn_dropout
        )
        
        # 保存配置
        self.use_l2norm = use_l2norm
        self.loss_name = loss_name
        self.gamma = gamma
        
        # 设置随机种子
        tf.random.set_seed(seed)

    def call(self, inputs):
        """
        模型前向传播
        
        参数:
        -----
        inputs : dict
            输入字典，包含:
            - 'user': 用户ID，shape=(batch_size,)
            - 'pos_item': 正样本物品ID，shape=(batch_size,)
            - 'neg_item': 负样本物品ID，shape=(batch_size, neg_num)
        
        返回:
        -----
        logits : Tensor
            预测得分，shape=(batch_size, 1+neg_num)
        
        处理流程:
        ---------
        ┌─────────────────────────────────────────────────────────────┐
        │  User Tower                    Item Tower                    │
        │  ───────────                   ───────────                   │
        │                                                              │
        │  1. user_id                    1. pos_item_id / neg_item_id │
        │       ↓                              ↓                       │
        │  2. Embedding                  2. Embedding                  │
        │       ↓                              ↓                       │
        │  3. User MLP                   3. Item MLP                   │
        │       ↓                              ↓                       │
        │  4. (L2 Norm)                  4. (L2 Norm)                  │
        │       ↓                              ↓                       │
        │  user_vector                   item_vector                   │
        │       │                              │                       │
        │       └──────────→ 内积 ←────────────┘                       │
        │                     ↓                                        │
        │                   score                                      │
        └─────────────────────────────────────────────────────────────┘
        """
        
        # ==================== 用户塔处理 ====================
        
        # Step 1: 获取用户嵌入向量
        # 输入: (batch_size,) -> 输出: (batch_size, embed_dim)
        user_info = self.user_embedding_table(tf.reshape(inputs['user'], [-1, ]))
        
        # ==================== 物品塔处理 ====================
        
        # Step 2: 获取正样本物品嵌入向量
        # 输入: (batch_size,) -> 输出: (batch_size, embed_dim)
        pos_info = self.item_embedding_table(tf.reshape(inputs['pos_item'], [-1, ]))
        
        # Step 3: 获取负样本物品嵌入向量
        # 输入: (batch_size, neg_num) -> 输出: (batch_size, neg_num, embed_dim)
        neg_info = self.item_embedding_table(inputs['neg_item'])
        
        # ==================== MLP变换 ====================
        
        # Step 4: 用户向量通过MLP
        # 输入: (batch_size, embed_dim) -> 输出: (batch_size, user_mlp[-1])
        user_info = self.user_mlp_layer(user_info)
        
        # Step 5: 正样本物品向量通过MLP
        # 输入: (batch_size, embed_dim) -> 输出: (batch_size, item_mlp[-1])
        pos_info = self.item_mlp_layer(pos_info)
        
        # Step 6: 负样本物品向量通过MLP
        # 输入: (batch_size, neg_num, embed_dim) -> 输出: (batch_size, neg_num, item_mlp[-1])
        neg_info = self.item_mlp_layer(neg_info)
        
        # ==================== L2归一化（可选） ====================
        
        if self.use_l2norm:
            # 归一化后，内积 = 余弦相似度，取值范围[-1, 1]
            user_info = tf.math.l2_normalize(user_info, axis=-1)
            pos_info = tf.math.l2_normalize(pos_info, axis=-1)
            neg_info = tf.math.l2_normalize(neg_info, axis=-1)
        
        # ==================== 计算相似度得分 ====================
        
        # Step 7: 计算正样本得分（用户向量与正样本向量的内积）
        # user_info: (batch_size, dim)
        # pos_info: (batch_size, dim)
        # 逐元素相乘后求和 -> (batch_size, 1)
        pos_scores = tf.reduce_sum(
            tf.multiply(user_info, pos_info),
            axis=-1, 
            keepdims=True
        )  # shape: (batch_size, 1)
        
        # Step 8: 计算负样本得分
        # user_info扩展: (batch_size, dim) -> (batch_size, 1, dim)
        # neg_info: (batch_size, neg_num, dim)
        # 广播乘法后求和 -> (batch_size, neg_num)
        neg_scores = tf.reduce_sum(
            tf.multiply(tf.expand_dims(user_info, axis=1), neg_info),
            axis=-1
        )  # shape: (batch_size, neg_num)
        
        # ==================== 计算损失 ====================
        
        # Step 9: 添加损失函数
        # get_loss会根据loss_name选择合适的损失函数
        self.add_loss(get_loss(pos_scores, neg_scores, self.loss_name, self.gamma))
        
        # Step 10: 拼接输出
        # 用于评估时计算排名
        logits = tf.concat([pos_scores, neg_scores], axis=-1)
        # shape: (batch_size, 1 + neg_num)
        
        return logits

    def summary(self):
        """打印模型结构摘要"""
        inputs = {
            'user': Input(shape=(), dtype=tf.int32),
            'pos_item': Input(shape=(), dtype=tf.int32),
            'neg_item': Input(shape=(1,), dtype=tf.int32)
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()


"""
====================================================================================
【DSSM工业应用流程】
====================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│                              离线阶段                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. 训练模型                                                                   │
│     ├── 准备训练数据 (user_id, pos_item_id, neg_item_ids)                     │
│     ├── 训练DSSM模型                                                          │
│     └── 保存模型参数                                                           │
│                                                                               │
│  2. 导出物品向量                                                               │
│     ├── 遍历所有物品ID                                                         │
│     ├── 通过物品塔计算所有物品向量                                              │
│     └── 保存物品向量到向量数据库（如Faiss, Milvus）                             │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              在线阶段                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  用户请求 ──→ 用户塔 ──→ 用户向量 ──→ 向量检索 ──→ Top-K候选物品              │
│                              │              │                                 │
│                              │              ↓                                 │
│                              │         物品向量库                              │
│                              │         (Faiss等)                              │
│                              │                                                │
│  时延要求: < 50ms                                                             │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘

====================================================================================
【使用示例】
====================================================================================

# 1. 创建模型
model = DSSM(
    user_num=10000,                    # 用户数量
    item_num=50000,                    # 物品数量
    embed_dim=64,                      # 初始嵌入维度
    user_mlp=[128, 64, 32],           # 用户塔MLP
    item_mlp=[128, 64, 32],           # 物品塔MLP（最后一层维度必须相同！）
    activation='relu',                 # 激活函数
    dnn_dropout=0.2,                   # Dropout
    use_l2norm=True,                   # 使用L2归一化
    loss_name='bpr_loss',              # 使用BPR损失
    embed_reg=1e-6                     # 正则化
)

# 2. 编译并训练
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(x=train_data, epochs=20, batch_size=512)

# 3. 导出物品向量（用于建立向量索引）
all_item_ids = np.arange(50000)
item_vectors = model.item_embedding_table(all_item_ids)
item_vectors = model.item_mlp_layer(item_vectors)
if model.use_l2norm:
    item_vectors = tf.math.l2_normalize(item_vectors, axis=-1)

# 4. 建立Faiss索引（示例）
import faiss
index = faiss.IndexFlatIP(32)  # 内积索引，维度32
index.add(item_vectors.numpy())

# 5. 在线召回
user_vector = model.user_embedding_table(np.array([user_id]))
user_vector = model.user_mlp_layer(user_vector)
if model.use_l2norm:
    user_vector = tf.math.l2_normalize(user_vector, axis=-1)
    
# 检索Top-K
scores, indices = index.search(user_vector.numpy(), k=100)

====================================================================================
"""

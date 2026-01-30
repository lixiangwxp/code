"""
====================================================================================
损失函数详解 (Loss Functions)
====================================================================================
创建日期: Nov 14, 2021
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

本文件包含推荐系统中常用的损失函数，用于训练召回模型：
1. BPR Loss - 贝叶斯个性化排序损失
2. Binary Cross Entropy Loss - 二元交叉熵损失
3. Hinge Loss - 合页损失

【损失函数的选择指南】
┌──────────────────────┬──────────────────────────────────────────────┐
│      损失函数         │              适用场景                          │
├──────────────────────┼──────────────────────────────────────────────┤
│ BPR Loss            │ 隐式反馈，成对学习（pairwise）                  │
│                     │ 目标：正样本得分 > 负样本得分                    │
├──────────────────────┼──────────────────────────────────────────────┤
│ Binary CE Loss      │ 点对学习（pointwise）                          │
│                     │ 目标：正样本预测为1，负样本预测为0               │
├──────────────────────┼──────────────────────────────────────────────┤
│ Hinge Loss          │ 成对学习，更关注margin                         │
│                     │ 目标：正样本得分比负样本高出至少gamma            │
└──────────────────────┴──────────────────────────────────────────────┘

【Pointwise vs Pairwise vs Listwise】

1. Pointwise（点对学习）:
   - 独立预测每个样本的分数/标签
   - 例如：预测用户是否会点击某物品
   - 损失函数：交叉熵、MSE

2. Pairwise（成对学习）:
   - 比较正负样本对的相对顺序
   - 例如：确保用户喜欢的物品得分更高
   - 损失函数：BPR、Hinge

3. Listwise（列表学习）:
   - 直接优化整个列表的排序
   - 例如：直接优化NDCG
   - 损失函数：ListNet、LambdaRank

推荐系统中常用Pairwise方法，因为我们更关心相对排序而非绝对分数。
====================================================================================
"""
import tensorflow as tf


def get_loss(pos_scores, neg_scores, loss_name, gamma=None):
    """
    根据名称获取对应的损失函数
    
    参数说明:
    ---------
    pos_scores : Tensor
        正样本得分，shape: (batch_size, 1) 或 (batch_size, neg_num)
        
    neg_scores : Tensor
        负样本得分，shape: (batch_size, neg_num)
        
    loss_name : str
        损失函数名称，可选:
        - 'bpr_loss': 贝叶斯个性化排序损失
        - 'hinge_loss': 合页损失
        - 其他: 默认使用二元交叉熵损失
        
    gamma : float, 可选
        hinge_loss的margin参数
    
    返回:
    ------
    loss : Tensor
        标量损失值
    
    使用示例:
    ---------
    >>> pos = tf.constant([[0.8], [0.7]])  # 正样本得分
    >>> neg = tf.constant([[0.3, 0.2], [0.4, 0.1]])  # 负样本得分
    >>> loss = get_loss(pos, neg, 'bpr_loss')
    """
    # 将正样本得分扩展到与负样本相同的维度
    # 这样可以对每个(正样本, 负样本)对计算损失
    # pos_scores: (batch_size, 1) -> (batch_size, neg_num)
    pos_scores = tf.tile(pos_scores, [1, neg_scores.shape[1]])
    
    # 根据名称选择损失函数
    if loss_name == 'bpr_loss':
        loss = bpr_loss(pos_scores, neg_scores)
    elif loss_name == 'hinge_loss':
        loss = hinge_loss(pos_scores, neg_scores, gamma)
    else:
        # 默认使用二元交叉熵损失
        loss = binary_cross_entropy_loss(pos_scores, neg_scores)
    
    return loss


def bpr_loss(pos_scores, neg_scores):
    """
    BPR损失（Bayesian Personalized Ranking Loss）
    
    【原理】
    BPR的目标是最大化正样本得分与负样本得分之差的对数似然。
    假设用户对正样本的偏好大于负样本，我们想最大化：
    P(正样本 > 负样本) = sigmoid(pos_score - neg_score)
    
    取负对数得到损失函数：
    loss = -log(sigmoid(pos_score - neg_score))
    
    【数学形式】
    L = -Σ log(σ(pos_i - neg_i))
    
    其中σ是sigmoid函数：σ(x) = 1 / (1 + e^(-x))
    
    【直观理解】
    - 当 pos_score > neg_score 时，sigmoid输出接近1，损失接近0
    - 当 pos_score < neg_score 时，sigmoid输出接近0，损失很大
    - 因此，模型会被训练使得正样本得分高于负样本得分
    
    参数说明:
    ---------
    pos_scores : Tensor
        正样本得分，shape: (batch_size, neg_num) 
        注意：已经tile过了，每行都是同一个正样本分数的复制
        
    neg_scores : Tensor
        负样本得分，shape: (batch_size, neg_num)
    
    返回:
    ------
    loss : Tensor
        标量损失值，是所有样本对损失的平均值
    
    计算过程可视化:
    ---------------
    pos_scores = [[0.8, 0.8, 0.8],    # 用户1的正样本得分（复制了3次）
                  [0.7, 0.7, 0.7]]    # 用户2的正样本得分
    
    neg_scores = [[0.3, 0.5, 0.2],    # 用户1的3个负样本得分
                  [0.4, 0.1, 0.6]]    # 用户2的3个负样本得分
    
    差值 = pos - neg = [[0.5, 0.3, 0.6],
                        [0.3, 0.6, 0.1]]
    
    sigmoid(差值) ≈ [[0.62, 0.57, 0.65],
                     [0.57, 0.65, 0.52]]
    
    -log(sigmoid) ≈ [[0.48, 0.56, 0.43],
                      [0.56, 0.43, 0.65]]
    
    loss = mean(上述矩阵) ≈ 0.52
    """
    # 计算正负样本得分之差
    diff = pos_scores - neg_scores
    
    # sigmoid激活后取负对数
    # tf.nn.sigmoid: 将差值映射到(0, 1)
    # tf.math.log: 取自然对数
    # 负号: 因为我们要最大化log(sigmoid)，等价于最小化-log(sigmoid)
    loss = tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(diff)))
    
    return loss


def hinge_loss(pos_scores, neg_scores, gamma=0.5):
    """
    Hinge损失（合页损失）
    
    【原理】
    Hinge Loss来源于SVM，在推荐系统中用于确保正样本得分
    比负样本得分高出至少一个margin（gamma）。
    
    【数学形式】
    L = Σ max(0, gamma - (pos_i - neg_i))
      = Σ max(0, neg_i - pos_i + gamma)
    
    【直观理解】
    - 当 pos_score - neg_score >= gamma 时，损失为0（已满足要求）
    - 当 pos_score - neg_score < gamma 时，损失为 gamma - (pos - neg)
    - gamma越大，要求正负样本分数差距越大，模型越难训练
    
    【与BPR的对比】
    ┌────────────────────┬─────────────────────┐
    │       BPR          │      Hinge          │
    ├────────────────────┼─────────────────────┤
    │ 平滑的损失曲线       │ 分段线性损失         │
    │ 总是有梯度          │ 满足margin后梯度为0  │
    │ 对所有样本都有惩罚   │ 只惩罚不满足margin的 │
    │ 适合一般场景        │ 适合需要明确间隔的   │
    └────────────────────┴─────────────────────┘
    
    参数说明:
    ---------
    pos_scores : Tensor
        正样本得分，shape: (batch_size, neg_num)
        
    neg_scores : Tensor
        负样本得分，shape: (batch_size, neg_num)
        
    gamma : float, 默认0.5
        margin参数，正样本得分需要比负样本高出至少gamma
        - gamma太小: 约束太弱，模型容易欠拟合
        - gamma太大: 约束太强，模型难以收敛
    
    返回:
    ------
    loss : Tensor
        标量损失值
    
    计算示例:
    ---------
    gamma = 0.5
    pos_scores = [[0.8, 0.8],
                  [0.7, 0.7]]
    neg_scores = [[0.3, 0.6],
                  [0.4, 0.7]]
    
    neg - pos + gamma = [[-0.0, 0.3],   # 0.3-0.8+0.5, 0.6-0.8+0.5
                          [0.2, 0.5]]   # 0.4-0.7+0.5, 0.7-0.7+0.5
    
    max(0, 上述) = [[0.0, 0.3],
                    [0.2, 0.5]]
    
    loss = mean = 0.25
    """
    # 计算 max(0, neg - pos + gamma)
    # tf.nn.relu: ReLU函数，即max(0, x)
    loss = tf.reduce_mean(tf.nn.relu(neg_scores - pos_scores + gamma))
    
    return loss


def binary_cross_entropy_loss(pos_scores, neg_scores):
    """
    二元交叉熵损失（Binary Cross Entropy Loss）
    
    【原理】
    将推荐问题转化为二分类问题：
    - 正样本的标签为1，希望预测接近1
    - 负样本的标签为0，希望预测接近0
    
    【数学形式】
    L = -[y*log(σ(s)) + (1-y)*log(1-σ(s))]
    
    对于正样本(y=1): L_pos = -log(σ(pos_score))
    对于负样本(y=0): L_neg = -log(1-σ(neg_score))
    
    总损失 = (L_pos + L_neg) / 2
    
    【直观理解】
    - 对于正样本：sigmoid(score)越接近1，损失越小
    - 对于负样本：sigmoid(score)越接近0，损失越小
    - 相当于希望正样本得分高（sigmoid接近1），负样本得分低（sigmoid接近0）
    
    【与BPR的对比】
    ┌─────────────────────┬──────────────────────┐
    │      BCE            │        BPR           │
    ├─────────────────────┼──────────────────────┤
    │ Pointwise学习       │ Pairwise学习          │
    │ 关注绝对分数        │ 关注相对排序           │
    │ 正负样本独立处理    │ 正负样本成对处理        │
    │ 适合CTR预估        │ 适合Top-K推荐          │
    └─────────────────────┴──────────────────────┘
    
    参数说明:
    ---------
    pos_scores : Tensor
        正样本得分，shape: (batch_size, neg_num)
        
    neg_scores : Tensor
        负样本得分，shape: (batch_size, neg_num)
    
    返回:
    ------
    loss : Tensor
        标量损失值
    
    计算示例:
    ---------
    pos_scores = [[0.8, 0.8],    # sigmoid后约为[0.69, 0.69]
                  [0.7, 0.7]]    # sigmoid后约为[0.67, 0.67]
    
    neg_scores = [[0.3, 0.5],    # sigmoid后约为[0.57, 0.62]
                  [0.4, 0.6]]    # sigmoid后约为[0.60, 0.65]
    
    正样本损失 = -log(sigmoid(pos)) ≈ [[0.37, 0.37], [0.40, 0.40]]
    负样本损失 = -log(1-sigmoid(neg)) ≈ [[0.84, 0.97], [0.92, 1.05]]
    
    总损失 = mean(正样本损失 + 负样本损失) / 2 ≈ 0.54
    """
    # 正样本损失: -log(sigmoid(pos_score))
    # 希望sigmoid(pos_score)接近1，即pos_score越大越好
    pos_loss = -tf.math.log(tf.nn.sigmoid(pos_scores))
    
    # 负样本损失: -log(1 - sigmoid(neg_score))
    # 希望sigmoid(neg_score)接近0，即neg_score越小越好
    neg_loss = -tf.math.log(1 - tf.nn.sigmoid(neg_scores))
    
    # 平均损失
    loss = tf.reduce_mean(pos_loss + neg_loss) / 2
    
    return loss


"""
====================================================================================
【损失函数可视化对比】
====================================================================================

假设pos_score固定为0，neg_score从-3变化到3

          Loss
            │
        4   │   ╲                              ← Hinge Loss (gamma=1)
            │    ╲                   
        3   │     ╲                   
            │      ╲                ╱╲         
        2   │       ╲         ╱────╱  ╲        ← BCE Loss
            │        ╲   ╱────                 
        1   │     ────────────────────         ← BPR Loss
            │                                   
        0   ├────────────────────────────────
            -3   -2   -1    0    1    2    3   neg_score
            
损失函数特性:
- BPR: 平滑曲线，始终有梯度，推动正负样本分数差距越来越大
- BCE: S形曲线，在中间区域梯度最大
- Hinge: 线性下降直到margin，之后梯度为0

====================================================================================
【使用建议】
====================================================================================

1. 如果目标是Top-K推荐（召回）:
   → 使用 BPR Loss 或 Hinge Loss
   → 因为我们关心的是排序，而非绝对分数

2. 如果目标是CTR预估（排序）:
   → 使用 Binary Cross Entropy Loss
   → 因为我们需要准确的点击概率估计

3. 如果负样本质量较高（hard negative）:
   → 使用 Hinge Loss，设置较小的gamma
   → 避免对已经正确排序的样本产生过多梯度

4. 如果训练不稳定:
   → 尝试 BPR Loss，它的梯度更平滑
   → 或调整Hinge Loss的gamma参数

====================================================================================
【代码使用示例】
====================================================================================

# 示例1: 使用BPR损失
pos_scores = model.predict_pos(inputs)  # (batch_size, 1)
neg_scores = model.predict_neg(inputs)  # (batch_size, neg_num)
loss = bpr_loss(tf.tile(pos_scores, [1, neg_num]), neg_scores)

# 示例2: 使用get_loss函数
loss = get_loss(pos_scores, neg_scores, 'bpr_loss')

# 示例3: 在模型中使用（推荐方式）
class MyModel(tf.keras.Model):
    def call(self, inputs):
        pos_scores = self.compute_score(inputs['pos_item'])
        neg_scores = self.compute_score(inputs['neg_item'])
        self.add_loss(get_loss(pos_scores, neg_scores, 'bpr_loss'))
        return tf.concat([pos_scores, neg_scores], axis=-1)

====================================================================================
"""

"""
====================================================================================
核心网络层详解 (Core Layers)
====================================================================================
创建日期: Nov 07, 2021
作者: Ziyao Geng(zggzy1996@163.com)
中文注释: RecLearn学习版

本文件包含推荐系统中常用的神经网络层组件，包括：
1. Linear - 线性层
2. MLP - 多层感知机
3. MultiHeadAttention - 多头注意力
4. TransformerEncoder - Transformer编码器
5. FM_Layer - FM因子分解机层
6. CrossNetwork - 交叉网络（DCN）
7. CIN - 压缩交互网络（xDeepFM）
8. CapsuleNetwork - 胶囊网络（MIND）

每个层都有详细的注释说明其输入输出和作用。
====================================================================================
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, LayerNormalization, Conv1D, ReLU
from tensorflow.keras.regularizers import l2


# ====================================================================================
# 1. Linear层 - 线性变换
# ====================================================================================
class Linear(Layer):
    """
    线性层（用于Wide&Deep、xDeepFM等模型的Wide部分）
    
    功能: 对输入特征进行线性加权求和
    公式: y = Σ wi * xi
    
    使用场景:
    - Wide&Deep的Wide部分
    - FM的一阶部分
    - xDeepFM的线性部分
    """
    
    def __init__(self, feature_length, w_reg=1e-6):
        """
        参数说明:
        ---------
        feature_length : int
            特征总数量（所有稀疏特征的取值个数之和）
            例如: user_id有10000个取值，item_id有50000个取值
                  则feature_length = 10000 + 50000 = 60000
            
        w_reg : float, 默认1e-6
            权重的L2正则化系数
        """
        super(Linear, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        """
        创建权重变量
        
        权重矩阵w的shape为(feature_length, 1)
        每个特征值对应一个权重
        """
        self.w = self.add_weight(
            name="w",
            shape=(self.feature_length, 1),  # 每个特征一个权重
            regularizer=l2(self.w_reg),
            trainable=True
        )

    def call(self, inputs):
        """
        前向传播
        
        输入: inputs - 特征索引，shape: (batch_size, field_num)
              例如: [[user_id, item_id, cate_id], ...]
        
        输出: 线性得分，shape: (batch_size, 1)
        
        处理流程:
        1. 通过embedding_lookup获取每个特征的权重
        2. 对所有权重求和
        """
        # tf.nn.embedding_lookup: 根据索引查表获取权重
        # 输入: (batch_size, field_num)
        # 输出: (batch_size, field_num, 1)
        weights = tf.nn.embedding_lookup(self.w, inputs)
        
        # 在axis=1上求和（所有特征的权重相加）
        # 输出: (batch_size, 1)
        result = tf.reduce_sum(weights, axis=1)
        
        return result


# ====================================================================================
# 2. MLP层 - 多层感知机
# ====================================================================================
class MLP(Layer):
    """
    多层感知机（Multilayer Perceptron）
    
    功能: 多层全连接神经网络，用于学习高阶特征交互
    
    使用场景:
    - DeepFM的Deep部分
    - Wide&Deep的Deep部分
    - DCN的Deep部分
    - DSSM的用户塔和物品塔
    - 几乎所有深度推荐模型
    
    网络结构:
    Input -> Dense -> Activation -> [Dropout] -> Dense -> ... -> Output
    """
    
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., use_batch_norm=False):
        """
        参数说明:
        ---------
        hidden_units : list
            隐藏层单元数列表
            例如: [256, 128, 64] 表示3层，单元数分别为256、128、64
            
        activation : str, 默认'relu'
            激活函数名称
            可选: 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'
            
        dnn_dropout : float, 默认0.0
            Dropout比例（0-1之间）
            训练时随机丢弃这个比例的神经元，防止过拟合
            
        use_batch_norm : bool, 默认False
            是否使用批量归一化
            可以加速训练，但在推荐系统中效果不一定好
        """
        super(MLP, self).__init__()
        
        # 创建多层Dense层
        # 每层后面都接激活函数
        self.dnn_network = [
            Dense(units=unit, activation=activation) 
            for unit in hidden_units
        ]
        
        # Dropout层
        self.dropout = Dropout(dnn_dropout)
        
        # 批量归一化
        self.use_batch_norm = use_batch_norm
        self.bt = BatchNormalization()

    def call(self, inputs):
        """
        前向传播
        
        输入: inputs - 特征向量，shape: (batch_size, input_dim)
        输出: 变换后的向量，shape: (batch_size, hidden_units[-1])
        
        处理流程:
        1. 依次通过每层Dense
        2. (可选) 批量归一化
        3. Dropout
        """
        x = inputs
        
        # 依次通过每层Dense
        for dnn in self.dnn_network:
            x = dnn(x)
        
        # 批量归一化（可选）
        if self.use_batch_norm:
            x = self.bt(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


# ====================================================================================
# 3. 多头注意力机制
# ====================================================================================
class MultiHeadAttention(Layer):
    """
    多头注意力机制（Multi-Head Attention）
    
    功能: Transformer的核心组件，能够让模型关注序列中不同位置的信息
    
    使用场景:
    - SASRec的自注意力层
    - Transformer Encoder
    
    核心公式:
    Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o
    
    多头的意义:
    - 每个头可以关注不同的信息
    - 增加模型的表达能力
    """
    
    def __init__(self, d_model, num_heads):
        """
        参数说明:
        ---------
        d_model : int
            模型维度（输入和输出的维度）
            必须能被num_heads整除
            
        num_heads : int
            注意力头的数量
            d_model / num_heads = 每个头的维度(depth)
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Q, K, V的线性变换层
        self.wq = Dense(d_model, activation=None)  # Query变换
        self.wk = Dense(d_model, activation=None)  # Key变换
        self.wv = Dense(d_model, activation=None)  # Value变换

    def call(self, q, k, v, mask):
        """
        前向传播
        
        输入:
        - q: Query张量，shape: (batch_size, seq_len, d_model)
        - k: Key张量，shape: (batch_size, seq_len, d_model)
        - v: Value张量，shape: (batch_size, seq_len, d_model)
        - mask: 掩码，shape: (batch_size, seq_len, 1)
        
        输出:
        - attention输出，shape: (batch_size, seq_len, d_model)
        
        处理流程:
        1. 对Q, K, V进行线性变换
        2. 分割成多个头
        3. 计算缩放点积注意力
        4. 拼接多头输出
        """
        # Step 1: 线性变换
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        seq_len, d_model = q.shape[1], q.shape[2]
        depth = d_model // self.num_heads
        
        # Step 2: 分割成多个头
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)
        q = self._split_heads(q, seq_len, depth)
        k = self._split_heads(k, seq_len, depth)
        v = self._split_heads(v, seq_len, depth)
        
        # 扩展mask以匹配多头维度
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_heads, 1, 1])
        
        # Step 3: 缩放点积注意力
        scaled_attention = self._scaled_dot_product_attention(q, k, v, mask)
        
        # Step 4: 重新拼接
        # (batch_size, num_heads, seq_len, depth) -> (batch_size, seq_len, d_model)
        outputs = tf.reshape(
            tf.transpose(scaled_attention, [0, 2, 1, 3]), 
            [-1, seq_len, d_model]
        )
        
        return outputs
    
    def _split_heads(self, x, seq_len, depth):
        """将最后一个维度分割成(num_heads, depth)"""
        x = tf.reshape(x, (-1, seq_len, self.num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def _scaled_dot_product_attention(self, q, k, v, mask):
        """
        缩放点积注意力
        
        公式: softmax(Q * K^T / sqrt(d_k)) * V
        """
        # Q * K^T
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # 缩放
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        scaled_attention_logits = matmul_qk / tf.sqrt(dk)
        
        # 应用mask
        paddings = tf.ones_like(scaled_attention_logits) * (-2 ** 32 + 1)
        scaled_attention_logits = tf.where(
            tf.equal(mask, 0), paddings, scaled_attention_logits
        )
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # 加权求和
        output = tf.matmul(attention_weights, v)
        
        return output


# ====================================================================================
# 4. Transformer编码器层
# ====================================================================================
class TransformerEncoder(Layer):
    """
    Transformer编码器层
    
    功能: Transformer的编码器部分，用于序列建模
    
    使用场景:
    - SASRec模型
    - 其他基于Transformer的序列推荐模型
    
    结构:
    Input
      ↓
    Multi-Head Attention
      ↓
    Add & LayerNorm (残差连接)
      ↓
    Feed Forward Network
      ↓
    Add & LayerNorm (残差连接)
      ↓
    Output
    """
    
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., layer_norm_eps=1e-6):
        """
        参数说明:
        ---------
        d_model : int
            模型维度
            
        num_heads : int, 默认1
            注意力头数量
            
        ffn_hidden_unit : int, 默认128
            前馈网络的隐藏层维度
            通常设为d_model的2-4倍
            
        dropout : float, 默认0.0
            Dropout比例
            
        layer_norm_eps : float, 默认1e-6
            LayerNorm的epsilon参数
        """
        super(TransformerEncoder, self).__init__()
        
        # 多头注意力
        self.mha = MultiHeadAttention(d_model, num_heads)
        
        # 前馈网络
        self.ffn = self._FFN(ffn_hidden_unit, d_model)
        
        # 层归一化
        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)
        
        # Dropout
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    
    def _FFN(self, hidden_unit, d_model):
        """前馈网络：两层线性变换"""
        return tf.keras.Sequential([
            Conv1D(filters=hidden_unit, kernel_size=1, activation='relu'),
            Conv1D(filters=d_model, kernel_size=1, activation=None)
        ])

    def call(self, inputs):
        """
        前向传播
        
        输入: [x, mask]
        - x: 序列嵌入，shape: (batch_size, seq_len, d_model)
        - mask: 掩码，shape: (batch_size, seq_len, 1)
        
        输出: 编码后的序列，shape: (batch_size, seq_len, d_model)
        """
        x, mask = inputs
        
        # 自注意力 + 残差连接 + LayerNorm
        att_out = self.mha(x, x, x, mask)  # 自注意力：Q=K=V
        att_out = self.dropout1(att_out)
        out1 = self.layernorm1(x + att_out)  # 残差连接
        
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)  # 残差连接
        
        return out2


# ====================================================================================
# 5. FM层 - 因子分解机核心层
# ====================================================================================
class FM_Layer(Layer):
    """
    FM层（Factorization Machines Layer）
    
    功能: 实现FM的一阶和二阶特征交叉
    
    使用场景:
    - FM模型
    - 作为其他模型的组件
    
    公式:
    y = w0 + Σ wi*xi + 1/2 * Σ[(Σ vi*xi)^2 - Σ(vi*xi)^2]
    
    第一部分: 全局偏置
    第二部分: 一阶线性
    第三部分: 二阶交叉（通过隐向量内积实现）
    """
    
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """
        参数说明:
        ---------
        feature_columns : list
            特征列配置列表
            
        k : int
            隐向量维度
            
        w_reg : float, 默认0.0
            一阶权重正则化系数
            
        v_reg : float, 默认0.0
            隐向量正则化系数
        """
        super(FM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(feature_columns)
        
        # 计算特征总数和建立映射
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        """创建权重"""
        # 全局偏置 w0，shape: (1,)
        self.w0 = self.add_weight(
            name='w0', 
            shape=(1,),
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        
        # 一阶权重 w，shape: (feature_length, 1)
        self.w = self.add_weight(
            name='w', 
            shape=(self.feature_length, 1),
            initializer=tf.random_normal_initializer(),
            regularizer=l2(self.w_reg),
            trainable=True
        )
        
        # 隐向量矩阵 V，shape: (feature_length, k)
        self.V = self.add_weight(
            name='V', 
            shape=(self.feature_length, self.k),
            initializer=tf.random_normal_initializer(),
            regularizer=l2(self.v_reg),
            trainable=True
        )

    def call(self, inputs):
        """
        前向传播
        
        输入: inputs - 特征字典 {'feat1': [...], 'feat2': [...], ...}
        输出: FM得分，shape: (batch_size, 1)
        
        处理流程:
        1. 特征索引映射
        2. 计算一阶部分
        3. 计算二阶部分（高效公式）
        4. 相加返回
        """
        # Step 1: 索引映射
        inputs = self._index_mapping(inputs)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        
        # Step 2: 一阶部分
        # w0 + Σ wi
        first_order = self.w0 + tf.reduce_sum(
            tf.nn.embedding_lookup(self.w, inputs), 
            axis=1
        )  # (batch_size, 1)
        
        # Step 3: 二阶部分
        # 获取每个特征的隐向量
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, k)
        
        # 高效计算公式:
        # 1/2 * [(Σ vi)^2 - Σ(vi^2)]
        
        # (Σ vi)^2
        square_sum = tf.square(
            tf.reduce_sum(second_inputs, axis=1, keepdims=True)
        )  # (batch_size, 1, k)
        
        # Σ(vi^2)
        sum_square = tf.reduce_sum(
            tf.square(second_inputs), 
            axis=1, 
            keepdims=True
        )  # (batch_size, 1, k)
        
        # 二阶交叉 = 1/2 * (square_sum - sum_square)
        second_order = 0.5 * tf.reduce_sum(
            square_sum - sum_square, 
            axis=2
        )  # (batch_size, 1)
        
        # Step 4: 输出
        outputs = first_order + second_order
        
        return outputs
    
    def _index_mapping(self, inputs_dict):
        """将特征值映射到全局索引"""
        outputs_dict = {}
        for key, value in inputs_dict.items():
            offset = self.map_dict.get(key, 0)
            outputs_dict[key] = tf.reshape(value + offset, [-1, 1])
        return outputs_dict


# ====================================================================================
# 6. 交叉网络（DCN的核心组件）
# ====================================================================================
class CrossNetwork(Layer):
    """
    交叉网络（Cross Network）
    
    功能: DCN模型的核心组件，显式地学习有界阶数的特征交叉
    
    使用场景:
    - DCN (Deep & Cross Network)
    
    公式:
    x_{l+1} = x_0 * x_l^T * w_l + b_l + x_l
    
    特点:
    - 显式特征交叉
    - 参数高效（每层只需要O(d)个参数）
    - 交叉阶数等于网络层数+1
    """
    
    def __init__(self, layer_num, reg_w=0., reg_b=0.):
        """
        参数说明:
        ---------
        layer_num : int
            交叉层数
            层数越多，捕获的特征交叉阶数越高
            通常设置为2-4层
            
        reg_w : float, 默认0.0
            权重w的正则化系数
            
        reg_b : float, 默认0.0
            偏置b的正则化系数
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        """创建每层的权重和偏置"""
        dim = int(input_shape[-1])
        
        # 每层都有一个权重向量w和偏置向量b
        self.cross_weights = [
            self.add_weight(
                name='w_' + str(i),
                shape=(dim, 1),  # 注意：是向量不是矩阵
                initializer='random_normal',
                regularizer=l2(self.reg_w),
                trainable=True
            )
            for i in range(self.layer_num)
        ]
        
        self.cross_bias = [
            self.add_weight(
                name='b_' + str(i),
                shape=(dim, 1),
                initializer='random_normal',
                regularizer=l2(self.reg_b),
                trainable=True
            )
            for i in range(self.layer_num)
        ]

    def call(self, inputs):
        """
        前向传播
        
        输入: inputs - 特征向量，shape: (batch_size, dim)
        输出: 交叉后的向量，shape: (batch_size, dim)
        
        处理流程:
        对于每一层l:
        x_{l+1} = x_0 * (x_l * w_l) + b_l + x_l
        
        其中x_0是原始输入，保持不变
        """
        # 增加一维便于矩阵运算
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0
        
        for i in range(self.layer_num):
            # x_l^T * w_l，得到标量（每个样本一个）
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, 1, 1)
            
            # x_0 * x_l1 + b_l + x_l
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        
        # 去掉多余的维度
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        
        return x_l


# ====================================================================================
# 7. CIN层（xDeepFM的核心组件）
# ====================================================================================
class CIN(Layer):
    """
    压缩交互网络（Compressed Interaction Network）
    
    功能: xDeepFM的核心组件，显式学习有界阶数的特征交叉
    
    使用场景:
    - xDeepFM模型
    
    特点:
    - 在特征维度上进行交互（不同于DCN在bit维度）
    - 类似于CNN的结构
    - 计算复杂度较高
    
    与Cross Network的区别:
    - CIN在特征向量层面交互
    - Cross Network在特征值层面交互
    """
    
    def __init__(self, cin_size, l2_reg=0.):
        """
        参数说明:
        ---------
        cin_size : list
            每层的输出维度列表
            例如: [128, 128] 表示2层CIN，每层输出128个特征图
            
        l2_reg : float, 默认0.0
            L2正则化系数
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        """创建CIN的卷积核"""
        self.embedding_nums = input_shape[1]  # 特征域数量
        self.field_nums = [self.embedding_nums] + self.cin_size
        
        # 创建每层的卷积核
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs):
        """
        前向传播
        
        输入: inputs - 嵌入矩阵，shape: (batch_size, field_num, embed_dim)
        输出: CIN输出，shape: (batch_size, sum(cin_size))
        """
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        
        # 分割embed_dim维度
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)
            
            # 外积
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)
            result_2 = tf.reshape(result_1, [dim, -1, self.embedding_nums * self.field_nums[idx]])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])
            
            # 卷积
            result_4 = tf.nn.conv1d(
                input=result_3, 
                filters=self.cin_W['CIN_W_' + str(idx)], 
                stride=1,
                padding='VALID'
            )
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])
            
            hidden_layers_results.append(result_5)
        
        # 拼接所有层的输出
        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result, axis=-1)
        
        return result


"""
====================================================================================
【层使用示例汇总】
====================================================================================

# 1. MLP使用
mlp = MLP(hidden_units=[256, 128, 64], activation='relu', dnn_dropout=0.2)
output = mlp(input_tensor)  # (batch_size, 64)

# 2. TransformerEncoder使用
encoder = TransformerEncoder(d_model=64, num_heads=4, ffn_hidden_unit=128)
output = encoder([input_tensor, mask])  # (batch_size, seq_len, 64)

# 3. FM_Layer使用
fm = FM_Layer(feature_columns, k=8)
output = fm(inputs)  # (batch_size, 1)

# 4. CrossNetwork使用
cross = CrossNetwork(layer_num=3)
output = cross(input_tensor)  # (batch_size, dim)

# 5. CIN使用
cin = CIN(cin_size=[128, 128])
output = cin(embed_matrix)  # (batch_size, 256)

====================================================================================
"""

"""
====================================================================================
AttRec 模型训练示例
====================================================================================
创建日期: Nov 20, 2021
更新日期: Apr 23, 2022
作者: Ziyao Geng(zggzy1996@163.com)

【模型简介】
AttRec (Attention-based Recommendation) 
是一种基于注意力机制的序列推荐模型。
改了这里



它结合了用户的长期兴趣和短期兴趣来进行推荐预测。

【模型核心思想】
1. 长期兴趣: 通过用户ID的嵌入向量表示
2. 短期兴趣: 通过自注意力机制对用户历史行为序列建模
3. 最终得分 = w * 短期兴趣得分 + (1-w) * 长期兴趣得分

【使用的数据集】
MovieLens-1M: 包含约100万条用户对电影的评分记录
====================================================================================
"""

# ====================================================================================
# 导入必要的库
# ====================================================================================
import os                                    # 操作系统接口，用于设置环境变量
from absl import flags, app                  # Google的命令行参数解析库（比argparse更强大）
from time import time                        # 时间模块，用于计算训练耗时
from tensorflow.keras.optimizers import Adam # Adam优化器，深度学习中最常用的优化器

# 导入RecLearn库的组件
from reclearn.models.matching import AttRec          # AttRec模型（注意力序列推荐）
from reclearn.data.datasets import movielens as ml   # MovieLens数据集处理工具
from reclearn.evaluator import eval_pos_neg          # 评估函数（计算HR、MRR、NDCG等指标）

# ====================================================================================
# 命令行参数配置
# ====================================================================================
# FLAGS对象用于存储所有命令行参数，可以在代码中通过FLAGS.xxx访问
FLAGS = flags.FLAGS

# 设置TensorFlow日志级别为2，只显示错误信息，屏蔽警告和信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'   # 如果有多GPU，可以指定使用哪个GPU

# ====================================================================================
# 定义训练参数（可以通过命令行修改）
# 格式: flags.DEFINE_类型("参数名", 默认值, "参数说明")
# ====================================================================================

# ----- 数据路径参数 -----
flags.DEFINE_string("file_path", "example/data/ml-1m/ratings.dat", 
    "原始数据文件路径，MovieLens-1M的评分数据")
flags.DEFINE_string("train_path", "None", 
    "训练集路径。如果设为None，程序会自动划分数据集")

flags.DEFINE_string("val_path", "data/ml-1m/ml_seq_val.txt", 
    "验证集路径，用于训练过程中监控模型性能")
flags.DEFINE_string("test_path", "data/ml-1m/ml_seq_test.txt", 
    "测试集路径，用于最终评估模型效果")
flags.DEFINE_string("meta_path", "data/ml-1m/ml_seq_meta.txt", 
    "元数据路径，存储用户和物品的最大索引")

# ----- 模型结构参数 -----
flags.DEFINE_integer("embed_dim", 64, 
    "嵌入向量维度。维度越大，模型表达能力越强，但也更容易过拟合")
flags.DEFINE_float("embed_reg", 0.0, 
    "嵌入层L2正则化系数。用于防止过拟合，常用值1e-6")
flags.DEFINE_string("mode", "inner", 
    "相似度计算方式: 'inner'=内积, 'dist'=欧氏距离")
flags.DEFINE_float("w", 0.3, 
    "短期兴趣权重。最终得分 = w*短期兴趣 + (1-w)*长期兴趣")
flags.DEFINE_boolean("use_l2norm", False, 
    "是否对嵌入向量进行L2归一化。归一化后内积等于余弦相似度")

# ----- 损失函数参数 -----
flags.DEFINE_string("loss_name", "hinge_loss", 
    "损失函数名称: 'hinge_loss'=合页损失, 'bpr_loss'=BPR损失, 'binary_cross_entropy_loss'=交叉熵")
flags.DEFINE_float("gamma", 0.5, 
    "Hinge Loss的margin参数。正样本得分需要比负样本高出至少gamma")

# ----- 训练参数 -----
flags.DEFINE_float("learning_rate", 0.001, 
    "学习率。控制每次参数更新的步长，太大容易震荡，太小收敛慢")
flags.DEFINE_integer("neg_num", 4, 
    "训练时每个正样本对应的负样本数量。负样本是随机采样的未交互物品")
flags.DEFINE_integer("seq_len", 100, 
    "用户行为序列长度。序列不足会padding，超过会截断")
flags.DEFINE_integer("epochs", 20, 
    "训练轮数。每轮会遍历一次完整的训练数据")
flags.DEFINE_integer("batch_size", 512, 
    "批次大小。每次梯度更新使用的样本数量")

# ----- 评估参数 -----
flags.DEFINE_integer("test_neg_num", 100, 
    "测试时每个正样本对应的负样本数量。用于计算排名")
flags.DEFINE_integer("k", 10, 
    "Top-K推荐的K值。计算HR@K、NDCG@K等指标")


def main(argv):
    """
    主函数：完整的模型训练和评估流程
    
    参数:
        argv: 命令行参数（由absl.app自动传入）
    
    训练流程:
        1. 数据划分 -> 2. 数据加载 -> 3. 模型配置 -> 4. 模型构建 -> 5. 训练评估
    """
    
    # ==================== 步骤1: 数据集划分 ====================
    # 如果train_path为"None"，则自动划分数据集
    # 否则使用指定的已划分好的数据集
    if FLAGS.train_path == "None":
        # 自动划分序列数据：按时间顺序，最后一个交互作为测试，倒数第二个作为验证
        train_path, val_path, test_path, meta_path = ml.split_seq_data(file_path=FLAGS.file_path)
    else:
        # 使用命令行指定的数据路径
        train_path, val_path, test_path, meta_path = FLAGS.train_path, FLAGS.val_path, FLAGS.test_path, FLAGS.meta_path
    
    # 读取元数据文件，获取用户和物品的最大索引
    # 元数据格式: "最大用户ID\t最大物品ID"
    with open(meta_path) as f:
        # 读取第一行，去除换行符，按tab分割，转换为整数
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    
    # ==================== 步骤2: 加载序列数据 ====================
    # load_seq_data函数会加载数据并生成负样本
    # 返回格式: {'user': [...], 'click_seq': [...], 'pos_item': [...], 'neg_item': [...]}
    
    # 加载训练数据
    # 参数: 文件路径, 模式("train"/"val"/"test"), 序列长度, 负样本数, 最大物品ID, 是否包含用户ID
    train_data = ml.load_seq_data(train_path, "train", FLAGS.seq_len, FLAGS.neg_num, max_item_num, contain_user=True)
    
    # 加载验证数据（用于训练时监控）
    val_data = ml.load_seq_data(val_path, "val", FLAGS.seq_len, FLAGS.neg_num, max_item_num, contain_user=True)
    
    # 加载测试数据（用于最终评估，负样本更多以便更准确地评估排名）
    test_data = ml.load_seq_data(test_path, "test", FLAGS.seq_len, FLAGS.test_neg_num, max_item_num, contain_user=True)
    
    # ==================== 步骤3: 设置模型超参数 ====================
    # 将所有模型参数打包成字典，便于传入模型
    model_params = {
        'user_num': max_user_num + 1,     # 用户数量（+1是因为ID从0开始）
        'item_num': max_item_num + 1,     # 物品数量
        'embed_dim': FLAGS.embed_dim,      # 嵌入维度
        'mode': FLAGS.mode,                # 相似度计算方式
        'w': FLAGS.w,                      # 短期兴趣权重
        'use_l2norm': FLAGS.use_l2norm,    # 是否L2归一化
        'loss_name': FLAGS.loss_name,      # 损失函数名称
        'gamma': FLAGS.gamma,              # Hinge Loss的margin
        'embed_reg': FLAGS.embed_reg       # 正则化系数
    }
    
    # ==================== 步骤4: 构建模型 ====================
    # 使用**model_params将字典解包为关键字参数
    model = AttRec(**model_params)
    
    # 编译模型：指定优化器
    # Adam优化器：自适应学习率优化器，结合了Momentum和RMSprop的优点
    model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
    
    # ==================== 步骤5: 训练和评估循环 ====================
    # 每个epoch：训练 -> 评估 -> 打印结果
    for epoch in range(1, FLAGS.epochs + 1):
        # 记录训练开始时间
        t1 = time()
        
        # 训练一个epoch
        # x: 训练数据
        # epochs=1: 每次只训练1轮（因为我们在外层循环控制）
        # validation_data: 验证数据，用于监控过拟合
        # batch_size: 批次大小
        model.fit(
            x=train_data,
            epochs=1,
            validation_data=val_data,
            batch_size=FLAGS.batch_size
        )
        
        # 记录训练结束时间
        t2 = time()
        
        # 在测试集上评估模型
        # 评估指标:
        # - HR (Hit Rate): 推荐列表中包含正确答案的比例
        # - MRR (Mean Reciprocal Rank): 正确答案排名倒数的均值
        # - NDCG (Normalized DCG): 考虑排名位置的归一化折损累积增益
        eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], FLAGS.k, FLAGS.batch_size)
        
        # 打印本轮训练结果
        # 包括：轮次、训练耗时、评估耗时、各项指标
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f'
              % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))


# ====================================================================================
# 程序入口
# ====================================================================================
if __name__ == '__main__':
    # 使用absl.app.run运行主函数
    # 它会自动解析命令行参数并传递给main函数
    app.run(main)


"""
====================================================================================
【运行示例】
====================================================================================

# 使用默认参数运行:
python m_attrec_demo.py

# 修改参数运行:
python m_attrec_demo.py --embed_dim=128 --epochs=30 --learning_rate=0.0001

# 查看所有可用参数:
python m_attrec_demo.py --help

====================================================================================
【数据格式说明】
====================================================================================

训练数据 (train_data) 格式:
{
    'user': [1, 2, 3, ...],                        # 用户ID, shape: (样本数,)
    'click_seq': [[1,2,3,...], [4,5,6,...], ...],  # 历史点击序列, shape: (样本数, seq_len)
    'pos_item': [10, 20, 30, ...],                 # 正样本物品ID, shape: (样本数,)
    'neg_item': [[5,6,7,8], [9,10,11,12], ...]     # 负样本物品ID, shape: (样本数, neg_num)
}

====================================================================================
【评估指标说明】
====================================================================================

HR@K (Hit Rate at K):
- 含义: 在Top-K推荐列表中，正确答案出现的比例
- 公式: HR = (正确推荐的用户数) / (总用户数)
- 范围: [0, 1]，越大越好

MRR@K (Mean Reciprocal Rank at K):
- 含义: 正确答案排名的倒数的平均值
- 公式: MRR = (1/N) * Σ (1/rank_i)
- 范围: [0, 1]，越大越好
- 例如: 如果正确答案排在第3位，则贡献1/3

NDCG@K (Normalized Discounted Cumulative Gain at K):
- 含义: 考虑排名位置的归一化增益
- 公式: NDCG = DCG / IDCG, 其中 DCG = Σ (2^rel - 1) / log2(rank + 1)
- 范围: [0, 1]，越大越好
- 特点: 对靠前位置的正确推荐给予更高的权重

====================================================================================
"""
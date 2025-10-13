
import jieba
from gensim.models import Word2Vec
import re

# 1. 训练支持类比推理的模型（使用更丰富的语义关联语料）
corpus = [
    "国王 男人 女王 女人",
    "父亲 儿子 母亲 女儿",
    "医生 病人 老师 学生",
    "北京 中国 华盛顿 美国",
    "大 小 长 短",
    "快 慢 高 低"
]
sentences = [text.split() for text in corpus]  # 空格分词
model = Word2Vec(sentences, vector_size=100, window=2, min_count=1, sg=1)

# 2. 类比推理函数
def word_analogy(a, b, c):
    """计算 a - b + c 的结果词"""
    try:
        # 核心：向量运算 a_vec - b_vec + c_vec
        result = model.wv.most_similar(positive=[a, c], negative=[b], topn=1)
        return result[0][0]  # 返回最可能的词
    except KeyError as e:
        return f"错误：词 {e} 不在模型词汇表中"
    except Exception as e:
        return f"计算失败：{e}"

# 3. 互动实验
print("🧪 语义类比实验室：验证 'a - b + c = ?'")
print("示例：国王 - 男人 + 女人 = 女王；北京 - 中国 + 美国 = 华盛顿")

# 学生输入（可修改这3个词进行实验）
a = input("请输入词a（如：国王）：")
b = input("请输入词b（如：男人）：")
c = input("请输入词c（如：女人）：")

# 计算并展示结果
result = word_analogy(a, b, c)
print(f"\n📝 推理结果：{a} - {b} + {c} = {result}")

# 4. 探索提示
print("\n🔍 尝试这些组合：")
print("1. 父亲 - 儿子 + 女儿 = ?")
print("2. 医生 - 病人 + 学生 = ?")
print("3. 大 - 小 + 短 = ?")

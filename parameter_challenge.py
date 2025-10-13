
import jieba
from gensim.models import Word2Vec
import re

# 1. 固定测试语料和评分标准
corpus = [
    "猫 狗 动物 宠物 哺乳动物",  # 预期关联：猫-狗、动物-宠物
    "苹果 香蕉 水果 食物 蔬菜",  # 预期关联：苹果-香蕉、水果-食物
    "计算机 手机 电子 设备 工具"   # 预期关联：计算机-手机、电子-设备
]
# 预处理（直接用空格分词简化步骤）
sentences = [text.split() for text in corpus]

# 2. 评分函数（越符合人类认知，分数越高）
def score_model(model):
    score = 0
    # 测试1：猫和狗的相似度（预期高）
    if "猫" in model.wv and "狗" in model.wv:
        score += model.wv.similarity("猫", "狗") * 10
    
    # 测试2：苹果和香蕉的相似度（预期高）
    if "苹果" in model.wv and "香蕉" in model.wv:
        score += model.wv.similarity("苹果", "香蕉") * 10
    
    # 测试3：计算机和手机的相似度（预期高）
    if "计算机" in model.wv and "手机" in model.wv:
        score += model.wv.similarity("计算机", "手机") * 10
    
    # 扣分项：苹果和猫的相似度（预期低）
    if "苹果" in model.wv and "猫" in model.wv:
        score -= model.wv.similarity("苹果", "猫") * 5
    
    return round(score, 2)

# 3. 学生参数配置区（核心互动点）
print("⚙️ 参数挑战赛：调整以下参数，让模型评分更高！")
vector_size = int(input("设置词向量维度（建议10-200）："))  # 例如：50
window = int(input("设置上下文窗口大小（建议2-10）："))      # 例如：3
sg = int(input("选择模型类型（0=CBOW，1=Skip-gram）："))   # 例如：1

# 4. 训练并评分
model = Word2Vec(
    sentences,
    vector_size=vector_size,
    window=window,
    min_count=1,
    sg=sg
)
final_score = score_model(model)
print(f"\n🏆 你的模型评分：{final_score}分（满分约30分）")

# 5. 提示与探索
if final_score > 25:
    print("🌟 优秀！你的参数配置很合理")
elif final_score > 15:
    print("👍 不错，试试增大window或调整模型类型？")
else:
    print("💡 建议：向量维度不宜过小（如<30），窗口大小影响上下文捕捉")

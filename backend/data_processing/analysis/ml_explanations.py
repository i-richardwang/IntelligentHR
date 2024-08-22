ML_TOOL_INFO = """
**🤖 机器学习建模工具**

这个工具允许您上传数据，选择目标变量和特征，然后使用随机森林或决策树分类器进行机器学习建模。

主要功能包括：
- 数据上传和预览
- 目标变量和特征选择
- 自定义模型参数设置
- 自动化的模型训练和优化
- 模型性能评估
- 特征重要性可视化
- 模型记录跟踪

该工具使用交叉验证和独立的测试集来评估模型性能，确保结果的可靠性。
"""

CONFUSION_MATRIX_EXPLANATION = """
混淆矩阵展示了模型在各个类别上的预测情况：

- 左上角：正确预测为负类的样本数（真负例，TN）
- 右上角：错误预测为正类的样本数（假正例，FP）
- 左下角：错误预测为负类的样本数（假负例，FN）
- 右下角：正确预测为正类的样本数（真正例，TP）

理想情况下，对角线上的数字（TN和TP）应该较大，而非对角线上的数字（FP和FN）应该较小。

这个矩阵可以帮助我们理解模型在哪些类别上表现较好或较差，从而针对性地改进模型或调整决策阈值。
"""

CLASSIFICATION_REPORT_EXPLANATION = """
分类报告提供了每个类别的详细性能指标：

- Precision（精确率）：预测为正例中实际为正例的比例
- Recall（召回率）：实际为正例中被正确预测的比例
- F1-score：精确率和召回率的调和平均数
- Support：每个类别的样本数量

'macro avg' 是所有类别的简单平均，'weighted avg' 是考虑了每个类别样本数量的加权平均。

这些指标可以帮助我们全面评估模型在各个类别上的表现，特别是在处理不平衡数据集时。
"""

FEATURE_IMPORTANCE_EXPLANATION = """
特征重要性图展示了模型中各个特征的相对重要性：

- 重要性得分反映了每个特征对模型预测的贡献程度。
- 得分越高，表示该特征在模型决策中的影响越大。
- 这个排序可以帮助我们识别最关键的预测因素。

注意事项：
- 特征重要性不表示因果关系，只反映预测能力。
- 高度相关的特征可能会分散重要性得分。
- 不同类型的模型可能会产生不同的特征重要性排序。
- 解释时应结合领域知识和其他分析方法。

利用特征重要性，我们可以：
1. 聚焦于最重要的特征，优化数据收集和处理。
2. 简化模型，可能去除不太重要的特征。
3. 获得对预测过程的洞察，提升模型可解释性。
4. 指导进一步的特征工程和选择。
"""
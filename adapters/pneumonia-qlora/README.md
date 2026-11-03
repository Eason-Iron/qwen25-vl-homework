# PneumoniaMNIST QLoRA 课程实验模型

**Built with Qwen.** 本目录包含真实训练得到的 LoRA 适配器，不包含基础模型；必须与 Qwen/Qwen2.5-VL-3B-Instruct 一起使用。

## 用途与限制

这是儿童胸片二分类的教学研究实验，类别为 NORMAL 和 PNEUMONIA。它使用低分辨率、有限规模的公开数据，不能用于临床诊断、筛查或治疗决策。

## 训练

- 基础模型：[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)，4 位 NF4 量化。
- 数据：[MedMNIST PneumoniaMNIST](https://github.com/MedMNIST/MedMNIST) 的 64×64 版本；官方训练划分中按固定随机种子抽取 256 张，正常和肺炎各 128 张。
- LLaMA-Factory 0.9.5，QLoRA rank 8、alpha 16，训练 q_proj/v_proj，冻结视觉塔和视觉投影层；2 个 epoch、64 个优化步骤。
- 输出 `0` 表示 NORMAL，`1` 表示 PNEUMONIA。使用训练时同一提示及输入像素配置，见 [任务代码](../../vl_workbench/tasks.py)和[训练配置](../../configs/pneumonia_qlora_numeric.yaml)。
- 权重 SHA-256：`53bd669d43eb1a719e0a28222adb9d19bed6a32a3549ac0051b7a4541c9d27c`。

适配器权重是训练输出的原样副本；为便于迁移，复制后的 `adapter_config.json` 仅将基础模型路径改为官方模型 ID。

## 评估

在 32 张类别均衡的官方验证集样本上，基础模型和适配器准确率分别为 50.0% 和 59.4%。在另外 64 张类别均衡的官方测试集样本上，两者准确率均为 50.0%；适配器的正常召回率为 62.5%，肺炎召回率为 37.5%。测试样本排除了 4 张曾用于冒烟检查的图片。样本量小、不是完整测试集；结果不支持声称模型优于基础模型。

详细预测、抽样种子、训练日志和实验比较见 [实验记录](../../reports/README.md)。

## 许可

基础模型及衍生适配器受 [Qwen Research License](../../licenses/Qwen-Research-License.txt) 约束，仅用于非商业研究或评估。出处和必要署名见 [NOTICE](../../NOTICE)。数据遵循其原始许可。

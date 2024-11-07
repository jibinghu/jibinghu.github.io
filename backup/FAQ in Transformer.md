### 如何理解 Transformers 中 FFNs 的作用？

- attention会混合多个token的信息来提取特征，但每个channel（feature dimension）保持独立。而FFN不混合token，而是混合不同的feature channel。两种计算操作不同的层面来提取特征，相得益彰。
- MLP-mixer这篇paper提出的抽象是最好的。类比到transformer，attention就是token-mixing，ffns就是channel-mixing
- 
![image](https://github.com/user-attachments/assets/d30a1264-c1b9-491f-bb7a-fc2538b88a2f)

---

### https://www.zhihu.com/question/596771388/answer/3459193587

这篇文章讲 transformer 架构的来源很清晰

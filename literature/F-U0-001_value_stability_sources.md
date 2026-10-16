# F-U0-001：价值失稳的直接相关来源

检索日期2026-10-16。由完整U0末期Q/TD loss增大、学生性能崩溃及固定真实样本上的纯任务价值负值触发。以下为原论文局部阅读和作者源码核对，没有运行这些框架或实施研究修改。

| 来源 | 已核对机制与条件 | 成本、边界与本项目关系 |
|---|---|---|
| [RLPD，Ball等，ICML2023](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf)，§4.1–4.3/B | 演示/在线各半、critic LayerNorm、高UTD与10critic；环境相关配置另列 | 与本项目演示混合最直接相关。归一化不等于绝对回报校准，也不保证候选排名正确。独立RLPD基线保留其额外计算；主U0不会因阅读而加入10critic。 |
| [DroQ，Hiraoka等，arXiv2110.02034v2](https://arxiv.org/html/2110.02034v2)，§3、附录F/G/H/I | 在小critic集合中组合dropout和LayerNorm；原实验UTD20，源码为Linear→Dropout→LayerNorm→ReLU | 附录讨论两者的交互和不同集合规模；只加LayerNorm不能声称复现完整DroQ。其actor使用critic平均，与主SAC的min不同，不能直接整段复制更新器。dropout需额外随机评估，当前尚无必要引入。 |
| [REDQ，Chen等，arXiv2101.05982v2](https://arxiv.org/html/2101.05982v2)，§3及作者更新代码 | 从更多critic随机取小子集构造min target，支撑高UTD；actor评价使用均值 | 适合核对RLPD集合来源及成本。其表格理论含误差独立等假设，不能变成本项目神经网络的不确定性保证。当前U0 UTD=.25，不能直接照搬高UTD结论。 |

固定源码：RLPD `c90fd4baf28c9c9ef40a81460a2e395092844f88`（MIT，LICENCE），DroQ `4c7adddbaf7573ff9256b16091a3bbdbd536aad4`（MIT），REDQ `7b5d1bff39291a57325a2836bd397a55728960bb`（MIT）。具体文件与SHA256见 value_stability_source_manifest.json。DroQ/REDQ保持只读缓存，未安装依赖，未拷入主工程。

证据限制：OpenReview PDF遇浏览器验证，改读作者列出的arXiv v2；ICLR年份由arXiv comments与作者README交叉核对，未声称OpenReview页面访问成功。DroQ网页表格的学习率显示异常，不能照抄；本项目学习率仍取锁定配置/源码。这里不把三项机制全部加入U0。先完成C续训与真实分支诊断，再写唯一主要改动、等预算对照及失败判据。

后续状态（2026-10-16）：上述为U1实施前的来源核对记录。C续训及两组prefix40真实F1现已完成，均没有可辨识候选排名；U1_CHANGE.md随后预登记“只对SAC critic隐藏层加LayerNorm”，CPU/真实GPU smoke已通过，完整效果尚待训练。DroQ/REDQ仍未安装或运行。再次直接打开其OpenReview PDF仍遇验证页；搜索索引可见原始PDF会议页眉，不把这种检索结果写成成功打开或精读全文。

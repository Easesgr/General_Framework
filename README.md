#### 目录

> 源码已在Github、Gitee开源
>
> Github: https://github.com/Easesgr/General_Framework
>
> Gitee:https://gitee.com/ease-i/general-framework

根据主函数`main.sh`作为主控制流程，来讲解这个框架，也可以作为本框架的参考文档

>一、参数与配置模块
>
>二、日志与模型检查点模块
>
>三、数据加载模块
>
>四、模型构建模块
>
>五、训练与测试模块
>
>六、主程序入口模块

🧠 总结架构图如下：

```pgsql
General_Framework/
│
├── main.py                 # 主程序入口，控制整体流程（参数加载、模型构建、训练/测试）
├── main.sh                 # 启动脚本，可用于批量运行或命令行调用 main.py
├── log.py                  # 日志记录模块，包括 Checkpoint 管理等
├── utils.py                # 工具函数与参数解析（如 AttrDict、get_args）
│
├── configs/                # 配置文件目录
│   └── default.yaml        # 默认的 YAML 配置文件，定义训练/模型参数
│
├── data/                   # 数据加载模块
│   ├── __init__.py         # 便于作为包导入
│   └── dataloder.py        # 自定义数据集类与 DataLoader 构建函数
│
├── model/                  # 模型构建模块
│   ├── __init__.py         # 便于作为包导入
│   └── model.py            # 主模型架构定义（如 LKDA 或其他网络结构）
│
├── loss/                   # 损失函数模块
│   └── loss.py             # 自定义损失函数（如 CharbonnierLoss、PerceptualLoss 等）
│
├── pre/                    # 预训练模型或辅助网络模块
│   └── vgg.py              # 定义用于感知损失的 VGG 网络或加载预训练模型
│
├── trainer/                # 训练/测试流程封装
│   └── train.py            # Trainer 类，封装 train()、testall() 等核心逻辑
│
├── logs/                   # 日志文件保存目录（如训练输出、记录文件）
│
├── runs/                   # 每次运行的输出目录，通常包含模型、日志、结果图等
    └── 2025-07-04_09-27-34 # 某次运行的具体输出目录（时间戳命名）

```

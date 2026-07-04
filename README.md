# IntelligentHR
Intelligent HR solutions for the data-driven enterprise.

智能HR助手是一个实验性的人力资源管理工具集，旨在探索AI技术在HR领域的应用潜力。
工具集涵盖了从数据处理、文本分析到决策支持的多个HR工作分析环节，致力于为人力资源管理提供全方位的智能化解决方案。

## 快速开始

数据存储全部改为**纯本地**:向量检索用 libSQL(SQLite 的向量分支)、关系型数据用
SQLite、简历文件落本地文件系统。**无需 Milvus / MySQL / MinIO 等任何独立 server**,
`pip install` 后即可离线运行。

1. 克隆仓库:
   ```
   git clone https://github.com/i-Richard-me/IntelligentHR
   cd IntelligentHR
   ```

2. 配置环境变量:
   复制 `.env.example` 为 `.env`,填写 LLM / Embedding 等 API 密钥。数据存储相关路径
   已有合理默认值(均在 `data/` 下),一般无需改动。

3. 本地直接运行(推荐):
   ```
   pip install -r requirements.txt
   streamlit run frontend/app.py --server.port=8510
   ```
   首次运行会自动在 `data/` 下创建所需的 SQLite/libSQL 数据文件与目录。

4. 访问应用:
   打开浏览器,访问 `http://localhost:8510`。

5. 可选:监控(Langfuse)
   如需链路监控,可用 Docker 启动 Langfuse(未配置时应用会优雅降级、照常运行):
   ```
   docker-compose --profile langfuse up --build
   ```
   然后在 `.env` 中填写 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`,访问
   `http://localhost:3000`。

6. 可选:整体容器化
   ```
   docker-compose up --build
   ```
   仅启动应用容器本身(数据仍是容器内的本地 SQLite/文件),访问 `http://localhost:8510`。

> 说明:向量检索、简历库、SQL 助手目标业务库均为本地文件,首次运行会按需自动创建。

### 初始化 demo 数据(可选)

部分功能需要预置样例数据后才能完整生效。项目内置一键初始化脚本,把各功能所需的业务库与
向量集合一次性灌好,数据源均已随仓库携带、运行时不依赖任何外部服务:

```
python -m tools.seed_demo_data          # 初始化全部三个域
```

覆盖三个域(可用 `--domain sql|cleaning|table-op` 单独初始化,默认 `all`):

| 域 | 初始化内容 |
|----|-----------|
| `sql` | SQL 助手业务样例库(`SQLBOT_DB_PATH`,默认 `data/sqlbot.db`)+ 向量集合 `table_descriptions`(选表)/ `term_descriptions`(术语映射)/ `query_examples`(少样本) |
| `cleaning` | 数据清洗实体标准化集合 `company_data` / `school_data`(公司/学校名称) |
| `table-op` | 智能表格操作集合 `tools_description`(由项目自身工具函数导出)/ `data_operation_examples`(少样本) |

- 业务库步骤不依赖任何 LLM / 网络;向量库步骤会调用已配置的 embedding provider 生成向量,
  故运行前需在 `.env` 中配好 Embedding 相关变量。
- 向量集合默认「已存在且非空则跳过」,`--overwrite` 可原地重建;`--skip-business` /
  `--skip-vectors` 可只跑其中一步。招聘业务数据量可通过 `--activities` / `--interviewers` /
  `--candidates` 调整。

## 贡献

我们欢迎并感谢任何形式的贡献。请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

## 行为准则

本项目采用了贡献者公约定义的行为准则，以营造一个开放和友好的社区环境。详情请见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 许可

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 免责声明

本项目处于实验阶段，主要用于学习和研究目的。在实际应用中使用时请谨慎，并自行承担相关风险。
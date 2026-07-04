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

### 初始化 SQL 助手 demo 数据(可选)

SQL 助手需要一个业务样例库(`SQLBOT_DB_PATH`,默认 `data/sqlbot.db`)以及若干向量集合
(选表用的 `table_descriptions`、术语映射用的 `term_descriptions`、少样本用的
`query_examples`)。项目内置一键初始化脚本,自动生成一套自洽的招聘 demo 数据并分别灌入
本地 SQLite 业务库与 libSQL 向量库:

```
python -m tools.seed_demo_data
```

- 业务库步骤不依赖任何 LLM / 网络;向量库步骤会调用已配置的 embedding provider 生成向量,
  故运行前需在 `.env` 中配好 Embedding 相关变量。
- 向量集合默认「已存在且非空则跳过」,`--overwrite` 可原地重建;`--skip-business` /
  `--skip-vectors` 可只跑其中一步。数据量可通过 `--activities` / `--interviewers` /
  `--candidates` 调整。

## 贡献

我们欢迎并感谢任何形式的贡献。请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目。

## 行为准则

本项目采用了贡献者公约定义的行为准则，以营造一个开放和友好的社区环境。详情请见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

## 许可

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

## 免责声明

本项目处于实验阶段，主要用于学习和研究目的。在实际应用中使用时请谨慎，并自行承担相关风险。
#!/usr/bin/env python3
"""
虚构简历数据生成器（确定性）。

为 TalentMatch（简历推荐）功能造一批**完全虚构**的候选人简历，不涉及任何真实个人
数据。固定随机种子（默认 42），每次运行产出完全一致的数据，便于复现与幂等重建。

产出的每份 ``resume_data`` 字典严格对齐两处消费方的读取口径，从而「写进去的正好是
功能读取的」：

- 关系型库 ``full_resume``（``resume_sql_storage.store_full_resume`` / ``get_full_resume``）：
  ``id`` / ``personal_info`` / ``education`` / ``work_experiences`` / ``project_experiences``
  / ``characteristics`` / ``experience_summary`` / ``skills_overview`` / ``resume_format`` /
  ``file_or_url``。
- 向量集合（``resume_vector_storage.store_resume_in_milvus`` 的拆分口径）：由
  ``seed_demo_data`` 的拍平器把上述嵌套结构展开为 personal_infos / educations /
  work_experiences / project_experiences / skills / raw_resume_texts 六个集合的行。

字段名与嵌套层级均按 ``resume_data_models`` 与两个 storage 模块的真实读取路径确定
（``personal_info`` 单数、``education`` 单数、``work_experiences``/``project_experiences``
复数；工作/项目的 responsibilities/details 为列表）。

为让「简历推荐」有可区分的检索结果，候选人覆盖若干**职业原型**（后端 / 算法 / 前端 /
产品 / 人力资源 / 财务分析），每个原型的个人简介、技能、岗位职责、项目内容、教育专业
与概述文本内部自洽，从而对 "找一个懂推荐系统的算法工程师""招聘经理" 等需求能给出合理排序。
"""

import random
from typing import Dict, List

# 固定演示上传时间：raw_resume_texts.upload_date 需确定性（避免每次运行值漂移）。
DEMO_UPLOAD_DATE = "2024-01-01 00:00:00"

_SURNAMES = [
    "王", "李", "张", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "马", "朱", "胡", "林", "何", "高", "郭", "罗",
]
_GIVEN = [
    "伟", "芳", "敏", "静", "丽", "强", "磊", "洋", "艳", "勇",
    "军", "杰", "娟", "涛", "明", "超", "霞", "刚", "晨", "子墨",
    "梓涵", "一鸣", "浩然", "雨欣", "思远", "宇轩", "欣怡", "睿",
]
_CITIES = [
    "北京", "上海", "深圳", "杭州", "广州", "成都", "南京", "武汉", "西安", "苏州",
]
_SCHOOLS = [
    "清华大学", "北京大学", "浙江大学", "上海交通大学", "复旦大学", "南京大学",
    "武汉大学", "华中科技大学", "西安交通大学", "哈尔滨工业大学", "中山大学",
    "四川大学", "东南大学", "同济大学", "北京航空航天大学",
]
# 通用（虚构/泛化）公司名池，避免把虚构候选人挂到真实公司名下。
_COMPANIES = [
    "星环智能", "云图科技", "远见网络", "恒信数据", "博岳信息", "锐捷软件",
    "联创科技", "华胜集团", "智联云", "擎天科技", "启明星辰信息", "天工数据",
    "融慧金服", "广联智造", "东方通达",
]
_DEGREES = ["本科", "硕士"]


# 每个职业原型：个人简介 / 技能池 / 教育专业 / 岗位（标题+职责池）/ 项目（名称+角色+详情池）
# / 三段概述（特点、经历、技能）。生成时在各池内确定性抽样，保证一份简历内部自洽。
_ARCHETYPES: List[Dict] = [
    {
        "key": "backend",
        "role": "后端开发工程师",
        "summary": "专注后端服务架构与高并发系统设计，擅长微服务拆分、数据库优化与稳定性治理",
        "skills": [
            "Java", "Go", "Spring Boot", "MySQL", "Redis", "Kafka",
            "微服务架构", "分布式系统", "Docker", "Kubernetes", "性能调优", "MongoDB",
        ],
        "majors": ["计算机科学与技术", "软件工程", "信息安全"],
        "positions": [
            ("高级后端开发工程师", [
                "负责核心交易系统的微服务拆分与重构，将单体应用改造为可独立部署的服务集群",
                "设计并落地基于 Redis 的多级缓存方案，接口平均响应时间下降 60%",
                "主导数据库分库分表改造，支撑日订单量从百万级增长到千万级",
            ]),
            ("后端开发工程师", [
                "使用 Spring Boot 开发用户中心与权限系统，服务日均调用量超过 2 亿次",
                "基于 Kafka 搭建异步消息链路，解耦下单、库存与结算模块",
                "负责线上服务的监控告警与故障排查，保障系统可用性达到 99.95%",
            ]),
            ("服务端研发工程师", [
                "参与支付网关的对接与风控规则引擎开发",
                "编写单元测试与集成测试，推动核心模块测试覆盖率提升至 80%",
                "使用 Docker 与 Kubernetes 完成服务容器化与灰度发布流程搭建",
            ]),
        ],
        "projects": [
            ("高并发订单中心重构", "后端负责人", [
                "牵头订单域领域建模与服务边界划分，输出微服务拆分方案",
                "引入分布式事务与幂等设计，解决超卖与重复下单问题",
            ]),
            ("统一缓存中间件", "核心开发", [
                "封装通用缓存客户端，支持本地缓存与分布式缓存二级结构",
                "实现缓存击穿、雪崩防护与热点 key 自动探测",
            ]),
        ],
        "characteristics": "技术扎实、注重系统稳定性，擅长在高并发场景下做架构权衡，具备较强的问题定位与攻坚能力。",
        "experience": "多年后端研发经验，主导过交易/订单核心系统的微服务化重构与性能优化，熟悉分布式系统全链路治理。",
        "skills_overview": "精通 Java/Go 后端开发与 Spring Boot 生态，熟练使用 MySQL/Redis/Kafka，具备微服务、容器化与性能调优的完整落地经验。",
    },
    {
        "key": "algo",
        "role": "算法工程师",
        "summary": "面向搜索推荐与机器学习建模，擅长特征工程、模型迭代与线上效果优化",
        "skills": [
            "Python", "机器学习", "深度学习", "推荐系统", "PyTorch", "TensorFlow",
            "特征工程", "Spark", "SQL", "自然语言处理", "A/B 测试", "召回排序",
        ],
        "majors": ["人工智能", "统计学", "计算机科学与技术", "应用数学"],
        "positions": [
            ("高级算法工程师", [
                "负责首页信息流推荐的召回与排序模型迭代，核心指标点击率提升 12%",
                "设计多目标排序模型，平衡点击、时长与多样性等业务目标",
                "搭建离线特征平台与实验评估体系，支撑算法快速迭代与 A/B 验证",
            ]),
            ("机器学习工程师", [
                "构建用户画像与行为序列特征，提升冷启动场景下的推荐效果",
                "使用 PyTorch 训练深度召回模型，优化向量检索的召回率与时延",
                "参与反作弊与异常流量识别模型的开发与上线",
            ]),
            ("数据挖掘工程师", [
                "基于 Spark 处理亿级用户行为日志，产出建模所需的训练样本",
                "负责搜索相关性排序的特征挖掘与模型调优",
            ]),
        ],
        "projects": [
            ("个性化推荐系统升级", "算法负责人", [
                "重构召回链路，引入双塔向量召回替换部分规则召回",
                "落地多目标精排模型，线上人均时长提升 8%",
            ]),
            ("智能搜索相关性优化", "核心算法", [
                "构建 query 理解与语义匹配模型，提升长尾查询的相关性",
                "设计在线学习流程，缩短模型更新周期到小时级",
            ]),
        ],
        "characteristics": "数据敏感、逻辑严谨，善于从业务指标反推建模思路，具备较强的模型落地与实验驱动优化能力。",
        "experience": "多年搜索推荐算法经验，主导过召回与排序模型的多轮迭代，熟悉从数据、特征、建模到线上评估的完整闭环。",
        "skills_overview": "熟练掌握 Python 与主流深度学习框架，精通推荐系统召回排序与特征工程，具备大规模数据处理与 A/B 实验能力。",
    },
    {
        "key": "frontend",
        "role": "前端开发工程师",
        "summary": "专注前端工程化与交互体验，擅长复杂中后台系统与组件化建设",
        "skills": [
            "JavaScript", "TypeScript", "React", "Vue", "Webpack", "Node.js",
            "前端工程化", "性能优化", "CSS", "可视化", "微前端", "单元测试",
        ],
        "majors": ["软件工程", "计算机科学与技术", "数字媒体技术"],
        "positions": [
            ("高级前端开发工程师", [
                "负责数据分析中台的前端架构设计与组件库建设，提升多团队研发效率",
                "推动微前端改造，将多个独立系统整合到统一工作台",
                "主导首屏性能优化，白屏时间降低 45%",
            ]),
            ("前端开发工程师", [
                "使用 React/TypeScript 开发企业级中后台管理系统",
                "封装通用业务组件与图表可视化模块，沉淀团队组件库",
                "搭建前端 CI 流程与自动化测试，保障发布质量",
            ]),
            ("Web 开发工程师", [
                "负责 B 端产品的页面开发与交互还原，对接后端接口联调",
                "优化打包体积与加载策略，改善弱网环境下的使用体验",
            ]),
        ],
        "projects": [
            ("企业级组件库建设", "前端负责人", [
                "从零搭建组件库工程体系，覆盖表单、表格、图表等高频场景",
                "建立文档站与按需加载机制，被 10+ 业务线复用",
            ]),
            ("数据可视化大屏", "核心开发", [
                "基于 Canvas 与图表库实现实时数据大屏，支撑运营监控",
                "优化大数据量渲染性能，保证秒级刷新流畅",
            ]),
        ],
        "characteristics": "注重用户体验与代码质量，工程化意识强，善于沉淀通用能力提升团队整体效率。",
        "experience": "多年前端研发经验，主导过中后台系统与组件库建设，熟悉前端工程化、性能优化与微前端实践。",
        "skills_overview": "精通 React/Vue 与 TypeScript，熟悉前端工程化与性能优化，具备组件库建设与数据可视化的完整经验。",
    },
    {
        "key": "product",
        "role": "产品经理",
        "summary": "面向 B 端与数据产品，擅长需求分析、方案设计与跨团队协作推动落地",
        "skills": [
            "需求分析", "产品设计", "用户研究", "数据分析", "原型设计", "Axure",
            "项目管理", "商业分析", "SQL", "竞品分析", "增长策略", "PRD 撰写",
        ],
        "majors": ["信息管理与信息系统", "工商管理", "计算机科学与技术", "市场营销"],
        "positions": [
            ("高级产品经理", [
                "负责数据分析平台的产品规划，从 0 到 1 搭建自助取数与看板能力",
                "梳理跨部门需求并输出产品路线图，协调研发、设计与运营推进落地",
                "通过数据驱动的迭代，将核心功能月活用户提升 35%",
            ]),
            ("产品经理", [
                "负责企业内部管理系统的需求调研、方案设计与 PRD 撰写",
                "组织用户访谈与可用性测试，持续优化产品体验",
                "推动版本迭代与上线验收，把控产品质量与节奏",
            ]),
            ("产品助理", [
                "参与竞品分析与市场调研，输出分析报告支持产品决策",
                "维护需求池与迭代排期，协助跟进开发进度",
            ]),
        ],
        "projects": [
            ("自助式数据分析平台", "产品负责人", [
                "定义自助取数、可视化看板与权限体系的产品方案",
                "推动平台在多业务线落地，显著降低取数人力成本",
            ]),
            ("HR 一体化管理系统", "产品经理", [
                "梳理招聘、绩效、组织人事的业务流程并设计产品方案",
                "协同研发完成核心模块上线，提升 HR 日常作业效率",
            ]),
        ],
        "characteristics": "业务理解深入、沟通协调能力强，善于把复杂需求拆解为可落地的产品方案并推动交付。",
        "experience": "多年 B 端/数据产品经验，主导过数据平台与管理系统从规划到落地的完整周期，擅长数据驱动的产品迭代。",
        "skills_overview": "具备扎实的需求分析与产品设计能力，熟练使用原型与数据分析工具，擅长跨团队协作与项目推进。",
    },
    {
        "key": "hr",
        "role": "人力资源经理",
        "summary": "深耕招聘与人力资源管理，擅长人才引进、组织发展与员工关系维护",
        "skills": [
            "招聘管理", "人才测评", "绩效管理", "组织发展", "员工关系", "薪酬福利",
            "培训体系", "人力资源规划", "劳动关系", "HRBP", "数据分析", "校园招聘",
        ],
        "majors": ["人力资源管理", "工商管理", "劳动与社会保障", "心理学"],
        "positions": [
            ("人力资源经理", [
                "负责公司整体招聘策略制定与执行，年度完成关键岗位招聘 200+ 人",
                "搭建结构化面试与人才测评体系，提升招聘质量与用人部门满意度",
                "推动绩效管理与人才盘点，支撑组织能力建设与梯队培养",
            ]),
            ("招聘主管", [
                "负责技术与产品序列的招聘全流程，管理多渠道招聘与雇主品牌建设",
                "组织校园招聘与社会招聘活动，优化招聘转化与到岗率",
                "分析招聘漏斗数据，持续改进招聘效率与成本",
            ]),
            ("HRBP", [
                "作为业务伙伴支持事业部的组织、人才与文化工作",
                "处理员工关系与劳动风险，推动团队敬业度提升",
            ]),
        ],
        "projects": [
            ("招聘体系升级项目", "项目负责人", [
                "重构从需求到 offer 的招聘流程，缩短平均招聘周期 30%",
                "引入人才测评与数据看板，实现招聘过程可量化管理",
            ]),
            ("人才梯队建设计划", "核心成员", [
                "设计关键岗位继任者计划与培养方案",
                "组织人才盘点与发展评估，输出组织能力诊断报告",
            ]),
        ],
        "characteristics": "亲和力强、原则性好，既懂业务又懂人，擅长在招聘与组织发展中平衡效率与体验。",
        "experience": "多年人力资源与招聘管理经验，主导过招聘体系升级与人才梯队建设，熟悉选育用留全流程。",
        "skills_overview": "精通招聘管理与人才测评，熟悉绩效、薪酬与组织发展体系，具备数据驱动的人力资源管理能力。",
    },
    {
        "key": "finance",
        "role": "财务分析师",
        "summary": "专注财务分析与经营决策支持，擅长财务建模、预算管理与经营数据洞察",
        "skills": [
            "财务分析", "财务建模", "预算管理", "成本控制", "Excel", "SQL",
            "BI 工具", "经营分析", "数据可视化", "报表编制", "现金流管理", "Python",
        ],
        "majors": ["财务管理", "会计学", "金融学", "经济学"],
        "positions": [
            ("高级财务分析师", [
                "负责公司经营分析与月度财务报告，为管理层决策提供数据支持",
                "搭建财务模型与预算体系，跟踪预算执行并输出偏差分析",
                "推动业财一体化，联合业务部门优化成本结构与投入产出",
            ]),
            ("财务分析师", [
                "编制经营分析报表，监控核心财务指标与业务健康度",
                "参与年度预算编制与滚动预测，管理各部门费用预算",
                "使用 BI 工具搭建财务看板，提升数据获取与分析效率",
            ]),
            ("财务专员", [
                "负责日常核算、费用审核与报表编制工作",
                "协助完成成本核算与利润分析，支持经营决策",
            ]),
        ],
        "projects": [
            ("经营分析看板建设", "分析负责人", [
                "设计核心经营指标体系并落地为自动化 BI 看板",
                "打通财务与业务数据，实现经营情况的实时可视化",
            ]),
            ("全面预算管理优化", "核心成员", [
                "重构预算编制与审批流程，提升预算准确性与时效",
                "建立预算执行监控机制，及时预警费用超支风险",
            ]),
        ],
        "characteristics": "严谨细致、数字敏感，善于从财务数据中发现经营问题并提出改进建议。",
        "experience": "多年财务分析经验，主导过经营分析体系与预算管理优化，熟悉业财融合与数据化经营分析。",
        "skills_overview": "精通财务分析与建模，熟练使用 Excel/SQL/BI 工具，具备预算管理与经营数据洞察的完整能力。",
    },
]


class ResumeDataGenerator:
    """确定性虚构简历生成器。"""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    # ---- 基础字段生成 ---- #
    def _name(self) -> str:
        return self._rng.choice(_SURNAMES) + self._rng.choice(_GIVEN)

    def _phone(self) -> str:
        return "13" + "".join(str(self._rng.randint(0, 9)) for _ in range(9))

    def _year(self, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def _education(self, archetype: Dict) -> List[Dict]:
        """1~2 段教育经历（含研究生时年份递增、专业同域）。"""
        count = self._rng.choice([1, 1, 2])
        grad_year = self._year(2013, 2021)
        edus: List[Dict] = []
        degrees = ["本科"] if count == 1 else ["本科", "硕士"]
        year = grad_year
        for degree in degrees:
            edus.append({
                "institution": self._rng.choice(_SCHOOLS),
                "degree": degree,
                "major": self._rng.choice(archetype["majors"]),
                "graduation_year": str(year),
            })
            year += self._rng.randint(2, 3)  # 硕士毕业年份晚于本科
        return edus

    def _work_experiences(self, archetype: Dict) -> List[Dict]:
        """2~3 段工作经历，时间倒序、含一段实习的概率。"""
        positions = self._rng.sample(archetype["positions"], k=min(3, len(archetype["positions"])))
        count = self._rng.choice([2, 2, 3])
        chosen = positions[:count]
        works: List[Dict] = []
        end_year = self._year(2023, 2024)
        for idx, (title, resp_pool) in enumerate(chosen):
            span = self._rng.randint(1, 3)
            start_year = end_year - span
            # 最早的一段有一定概率标记为实习
            is_intern = idx == len(chosen) - 1 and self._rng.random() < 0.3
            works.append({
                "company": self._rng.choice(_COMPANIES),
                "position": title,
                "experience_type": "实习" if is_intern else "正式",
                "start_date": f"{start_year}-{self._rng.randint(1, 9):02d}",
                "end_date": ("至今" if idx == 0 and self._rng.random() < 0.4
                             else f"{end_year}-{self._rng.randint(1, 12):02d}"),
                "responsibilities": self._rng.sample(resp_pool, k=min(len(resp_pool), self._rng.choice([2, 3]))),
            })
            end_year = start_year  # 下一段更早
        return works

    def _project_experiences(self, archetype: Dict) -> List[Dict]:
        """1~2 段项目经历。"""
        projects = self._rng.sample(archetype["projects"], k=min(len(archetype["projects"]), self._rng.choice([1, 2])))
        projs: List[Dict] = []
        end_year = self._year(2022, 2024)
        for name, role, detail_pool in projects:
            start_year = end_year - self._rng.randint(0, 1)
            projs.append({
                "name": name,
                "role": role,
                "start_date": f"{start_year}-{self._rng.randint(1, 6):02d}",
                "end_date": f"{end_year}-{self._rng.randint(7, 12):02d}",
                "details": list(detail_pool),
            })
            end_year = start_year - 1
        return projs

    def _skills(self, archetype: Dict) -> List[str]:
        return self._rng.sample(archetype["skills"], k=self._rng.randint(6, 8))

    def _raw_text(self, resume_data: Dict) -> str:
        """把各段拼成一份可读的简历原文（供 raw_resume_texts 向量化）。"""
        pi = resume_data["personal_info"]
        lines = [
            f"姓名：{pi['name']}",
            f"求职意向：{resume_data['_role']}",
            f"个人简介：{pi['summary']}",
            "",
            "教育背景：",
        ]
        for e in resume_data["education"]:
            lines.append(f"  {e['graduation_year']}届 {e['institution']} {e['major']} {e['degree']}")
        lines.append("")
        lines.append("工作经历：")
        for w in resume_data["work_experiences"]:
            lines.append(f"  {w['start_date']} ~ {w['end_date']} {w['company']} {w['position']}（{w['experience_type']}）")
            for r in w["responsibilities"]:
                lines.append(f"    - {r}")
        if resume_data.get("project_experiences"):
            lines.append("")
            lines.append("项目经历：")
            for p in resume_data["project_experiences"]:
                lines.append(f"  {p['name']}（{p['role']}）")
                for d in p["details"]:
                    lines.append(f"    - {d}")
        lines.append("")
        lines.append("专业技能：" + "、".join(pi["skills"]))
        return "\n".join(lines)

    def resume(self, index: int) -> Dict:
        """生成单份虚构简历（``resume_data`` 完整结构）。"""
        archetype = _ARCHETYPES[index % len(_ARCHETYPES)]
        years_exp = self._rng.randint(2, 12)
        resume_id = f"resume-demo-{index:03d}"

        personal_info = {
            "name": self._name(),
            "email": f"{resume_id}@example.com",
            "phone": self._phone(),
            "address": self._rng.choice(_CITIES),
            "summary": f"{archetype['summary']}，拥有约 {years_exp} 年相关经验。",
            "skills": self._skills(archetype),
        }

        resume_data = {
            "id": resume_id,
            "_role": archetype["role"],  # 仅内部用于拼 raw_text，不入库
            "personal_info": personal_info,
            "education": self._education(archetype),
            "work_experiences": self._work_experiences(archetype),
            "project_experiences": self._project_experiences(archetype),
            "characteristics": archetype["characteristics"],
            "experience_summary": archetype["experience"],
            "skills_overview": archetype["skills_overview"],
            "resume_format": "structured",
            "file_or_url": f"{resume_id}.pdf",
            "upload_date": DEMO_UPLOAD_DATE,
        }
        resume_data["raw_text"] = self._raw_text(resume_data)
        return resume_data

    def resumes(self, count: int = 30) -> List[Dict]:
        """生成 ``count`` 份虚构简历（确定性、覆盖全部职业原型）。"""
        return [self.resume(i) for i in range(count)]


if __name__ == "__main__":
    import json

    data = ResumeDataGenerator().resumes(count=6)
    print(f"生成 {len(data)} 份虚构简历，样例：")
    sample = {k: v for k, v in data[0].items() if k != "_role"}
    print(json.dumps(sample, ensure_ascii=False, indent=2))

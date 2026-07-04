"""
公司 / 学校名称标准化 demo 数据（迁移自 standalone LLM-DataCleaner）。

用于数据清洗功能的向量集合种子：``company_data`` / ``school_data``。两个集合的
schema 均为 ``original_name``（向量字段）+ ``standard_name``（见 collections_config），
故此处每条数据是「原始名 -> 标准名」的二元组，覆盖全称 / 简称 / 曾用名 / 中英文别名
等多种写法映射到同一标准实体，供实体标准化时做向量近邻匹配。
"""

from typing import Dict, List

# （原始名, 标准名）
COMPANY_DEMO = [
    ("阿里巴巴集团控股有限公司", "阿里巴巴"),
    ("淘宝（中国）软件有限公司", "阿里巴巴"),
    ("浙江天猫技术有限公司", "阿里巴巴"),
    ("支付宝（杭州）信息技术有限公司", "阿里巴巴"),
    ("深圳市腾讯计算机系统有限公司", "腾讯"),
    ("腾讯科技（深圳）有限公司", "腾讯"),
    ("财付通支付科技有限公司", "腾讯"),
    ("北京百度网讯科技有限公司", "百度"),
    ("百度在线网络技术（北京）有限公司", "百度"),
    ("北京字节跳动科技有限公司", "字节跳动"),
    ("字节跳动有限公司", "字节跳动"),
    ("北京抖音信息服务有限公司", "字节跳动"),
    ("华为技术有限公司", "华为"),
    ("华为终端有限公司", "华为"),
    ("北京京东世纪贸易有限公司", "京东"),
    ("京东科技控股股份有限公司", "京东"),
    ("北京小米科技有限责任公司", "小米"),
    ("小米通讯技术有限公司", "小米"),
    ("杭州网易云音乐科技有限公司", "网易"),
    ("网易（杭州）网络有限公司", "网易"),
    ("北京美团科技有限公司", "美团"),
    ("上海拼多多网络科技有限公司", "拼多多"),
    ("滴滴出行科技有限公司", "滴滴"),
    ("北京嘀嘀无限科技发展有限公司", "滴滴"),
    ("微软（中国）有限公司", "微软"),
    ("Microsoft Corporation", "微软"),
    ("Google LLC", "谷歌"),
    ("谷歌信息技术（中国）有限公司", "谷歌"),
    ("Apple Inc.", "苹果"),
    ("苹果电子产品商贸（北京）有限公司", "苹果"),
    ("Amazon.com, Inc.", "亚马逊"),
    ("亚马逊（中国）投资有限公司", "亚马逊"),
    ("Meta Platforms, Inc.", "Meta"),
    ("Samsung Electronics Co., Ltd.", "三星"),
    ("三星（中国）投资有限公司", "三星"),
    ("Sony Group Corporation", "索尼"),
    ("索尼（中国）有限公司", "索尼"),
]

SCHOOL_DEMO = [
    ("北京大学", "北京大学"),
    ("Peking University", "北京大学"),
    ("PKU", "北京大学"),
    ("北大", "北京大学"),
    ("清华大学", "清华大学"),
    ("Tsinghua University", "清华大学"),
    ("清华大学深圳研究生院", "清华大学"),
    ("复旦大学", "复旦大学"),
    ("Fudan University", "复旦大学"),
    ("上海交通大学", "上海交通大学"),
    ("Shanghai Jiao Tong University", "上海交通大学"),
    ("浙江大学", "浙江大学"),
    ("Zhejiang University", "浙江大学"),
    ("南京大学", "南京大学"),
    ("Nanjing University", "南京大学"),
    ("中国科学技术大学", "中国科学技术大学"),
    ("中科大", "中国科学技术大学"),
    ("武汉大学", "武汉大学"),
    ("华中科技大学", "华中科技大学"),
    ("中山大学", "中山大学"),
    ("哈尔滨工业大学", "哈尔滨工业大学"),
    ("哈工大", "哈尔滨工业大学"),
    ("哈尔滨工业大学（深圳）", "哈尔滨工业大学"),
    ("北京航空航天大学", "北京航空航天大学"),
    ("北航", "北京航空航天大学"),
    ("同济大学", "同济大学"),
    ("西安交通大学", "西安交通大学"),
    ("MIT", "麻省理工学院"),
    ("Massachusetts Institute of Technology", "麻省理工学院"),
    ("Stanford University", "斯坦福大学"),
    ("Harvard University", "哈佛大学"),
    ("University of Cambridge", "剑桥大学"),
    ("University of Oxford", "牛津大学"),
    ("The University of Tokyo", "东京大学"),
    ("National University of Singapore", "新加坡国立大学"),
    ("NUS", "新加坡国立大学"),
]


def _rows(pairs) -> List[Dict]:
    """把（原始名, 标准名）二元组列表转成集合可插入的行字典。"""
    return [{"original_name": o, "standard_name": s} for o, s in pairs]


def company_rows() -> List[Dict]:
    """公司名称标准化种子行（company_data 集合）。"""
    return _rows(COMPANY_DEMO)


def school_rows() -> List[Dict]:
    """学校名称标准化种子行（school_data 集合）。"""
    return _rows(SCHOOL_DEMO)

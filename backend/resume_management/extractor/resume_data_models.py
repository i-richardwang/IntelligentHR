from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class PersonalInfo(BaseModel):
    """个人基本信息、概述/总结和技能的数据模型。"""

    name: str = Field(..., description="姓名")
    email: Optional[str] = Field(None, description="电子邮件地址")
    phone: Optional[str] = Field(None, description="电话号码")
    address: Optional[str] = Field(None, description="地址")
    summary: Optional[str] = Field(
        None, description="个人简介、总结或概述等(来自简历提取，而非生成总结)"
    )
    skills: Optional[List[str]] = Field(None, description="技能列表")


class Education(BaseModel):
    """教育经历数据模型。"""

    institution: str = Field(..., description="教育机构名称")
    degree: str = Field(..., description="学位")
    major: str = Field(..., description="专业")
    graduation_year: str = Field(..., description="毕业年份，仅填写年份数字")


class WorkExperience(BaseModel):
    """工作经历数据模型。"""

    company: str = Field(..., description="公司名称")
    position: str = Field(..., description="职位名称")
    start_date: str = Field(..., description="工作开始时间")
    end_date: str = Field(..., description="工作结束时间")
    responsibilities: List[str] = Field(
        ..., description="工作职责和成就等。**包含原文中的全部相关信息**"
    )
    experience_type: Literal["正式", "实习"] = Field(
        ..., description="工作经历类型（正式或实习）"
    )


class ProjectExperience(BaseModel):
    """项目经历数据模型（仅针对简历中单独列出的项目经历，而非从工作经历中提取）。"""

    name: str = Field(..., description="项目名称")
    role: str = Field(..., description="在项目中担任的角色")
    start_date: str = Field(..., description="项目开始时间")
    end_date: str = Field(..., description="项目结束时间")
    details: List[str] = Field(
        ...,
        description="项目详情，包括描述、职责和成就等。**包含原文中的全部相关信息**",
    )


class ResumeSummary(BaseModel):
    """简历概述数据模型。"""

    characteristics: str = Field(..., description="员工特点概述")
    experience: str = Field(..., description="工作和项目经历概述")
    skills_overview: str = Field(..., description="技能概述")


class Summary(BaseModel):
    """简历概述数据模型。"""

    summary: ResumeSummary = Field(..., description="简历概述")


class ResumePersonalEducation(BaseModel):
    """个人信息和教育背景数据模型。"""

    personal_info: PersonalInfo = Field(..., description="个人基本信息、概述和技能")
    education: List[Education] = Field(..., description="教育背景列表")


class ResumeWorkProject(BaseModel):
    """工作经历和项目经历数据模型。"""

    work_experiences: List[WorkExperience] = Field(..., description="工作经历列表")
    project_experiences: Optional[List[ProjectExperience]] = Field(
        None, description="项目经历列表"
    )

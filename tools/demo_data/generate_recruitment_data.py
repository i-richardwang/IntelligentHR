"""
招聘 demo 数据生成器（迁移自 standalone QueryBot，适配 IntelligentHR）。

生成一套自洽的虚构招聘业务数据，用于让 SQL 助手「开箱即跑」：

- 三张业务表（招聘活动 / 面试官 / 候选人），落到本地 SQLite 业务库；
- 三份向量元数据（表描述 / 术语映射 / 查询示例），灌入本地 libSQL 向量库，
  分别对应 ``table_descriptions`` / ``term_descriptions`` / ``query_examples`` 集合。

相较 QueryBot 原版的关键适配：

1. **纯内存**：各方法直接返回 ``List[Dict]``，不再落 CSV，由上层
   ``tools.seed_demo_data`` 直接导入 SQLite / libSQL，减少中间产物；
2. **数据自洽**：面试官、候选人的 ``activity_name`` 改为与活动表共享同一
   具名活动池（原版用 ``"Recruitment Activity N"``，导致跨表无法关联、
   few-shot 示例查不到数据）；
3. **字段对齐 IHR 消费方**：``table_descriptions`` 去掉 IHR 不消费的 ``schema``
   列（表结构由 ``table_structure_node`` 直接自省实时库获得）；
4. **SQL 方言**：``query_examples`` 的 SQL 从 MySQL 改写为 SQLite（去掉
   ``STR_TO_DATE`` 等 MySQL 专有函数；文本日期 ``YYYY-MM-DD`` 直接按字典序比较）。

``random.seed(42)`` 固定随机种子，保证每次生成结果可复现。
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List

# 固定随机种子，保证 demo 数据可复现
random.seed(42)


class RecruitmentDataGenerator:
    """招聘数据生成器（内存版）。"""

    def __init__(self):
        # 具名招聘活动池：三张业务表共享，保证跨表可关联
        self.activity_names = [
            "Spring Campus Recruitment", "Summer Internship Program",
            "Fall General Recruitment", "Annual Core Talent Hunt",
            "Technical Specialist Hiring", "Product Team Recruitment",
            "Management Trainee Program", "Executive Search Campaign",
            "New Graduate Recruitment", "Senior Engineer Hiring",
            "Global Talent Acquisition", "Employee Referral Drive",
        ]

        self.positions = [
            "Java Developer", "Python Developer", "Frontend Developer",
            "Android Developer", "iOS Developer", "QA Engineer",
            "DevOps Engineer", "Data Analyst", "Product Manager",
            "UI Designer", "Project Manager", "Software Architect",
            "Algorithm Engineer", "Operations Specialist", "HR Specialist",
        ]

        self.departments = [
            "Engineering", "Product Design", "Quality Assurance", "DevOps",
            "Data Science", "Human Resources", "Marketing", "Operations",
            "Finance", "Administration",
        ]

        self.companies = [
            "TechCorp", "InnovateLab", "DataFlow Inc", "CloudTech", "MobileSoft",
            "SmartSystems", "NextGen Tech", "DigitalWorks", "FutureTech", "CodeCraft",
        ]

        self.cities = [
            "New York", "San Francisco", "Seattle", "Austin", "Boston",
            "Chicago", "Denver", "Atlanta", "Portland", "San Diego",
        ]

        self.job_levels = [
            "Junior", "Mid-Level", "Senior", "Staff", "Principal",
            "Manager", "Senior Manager", "Director",
        ]

        self.first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard",
            "Joseph", "Thomas", "Christopher", "Charles", "Daniel", "Matthew",
            "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
        ]

        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        ]

    # ------------------------------------------------------------------ #
    # 通用辅助
    # ------------------------------------------------------------------ #
    def generate_name(self) -> str:
        """生成随机姓名。"""
        return f"{random.choice(self.first_names)} {random.choice(self.last_names)}"

    def generate_employee_id(self, prefix: str = "EMP") -> str:
        """生成员工/候选人编号。"""
        return f"{prefix}{random.randint(100000, 999999)}"

    def generate_email(self, name: str, company: str = "company") -> str:
        """生成邮箱地址。"""
        return f"{name.lower().replace(' ', '.')}.{random.randint(100, 999)}@{company.lower()}.com"

    # ------------------------------------------------------------------ #
    # 业务表
    # ------------------------------------------------------------------ #
    def generate_recruitment_activities(self, count: int = 50) -> List[Dict]:
        """生成招聘活动数据。"""
        activities = []
        for i in range(count):
            start_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300))
            end_date = start_date + timedelta(days=random.randint(7, 60))
            activities.append({
                "id": i + 1,
                "activity_number": f"REC2024{str(i + 1).zfill(3)}",
                "activity_name": random.choice(self.activity_names),
                "position_type": random.choice(self.positions),
                "department": random.choice(self.departments),
                "recruitment_start_date": start_date.strftime("%Y-%m-%d"),
                "recruitment_end_date": end_date.strftime("%Y-%m-%d"),
                "recruitment_city": random.choice(self.cities),
                "target_headcount": random.randint(5, 50),
                "received_resumes": random.randint(20, 200),
                "screened_resumes": random.randint(10, 100),
                "interview_candidates": random.randint(5, 50),
                "offer_count": random.randint(1, 20),
                "onboard_count": random.randint(1, 15),
                "success_rate": round(random.uniform(0.1, 0.8), 2),
                "avg_interview_score": round(random.uniform(3.0, 4.5), 1),
                "hr_satisfaction": round(random.uniform(3.5, 5.0), 1),
                "hiring_manager_satisfaction": round(random.uniform(3.0, 5.0), 1),
                "job_level_requirement": random.choice(self.job_levels),
                "min_experience_years": random.randint(0, 8),
                "max_experience_years": random.randint(3, 15),
            })
        return activities

    def generate_interviewers(self, count: int = 100) -> List[Dict]:
        """生成面试官数据。"""
        interview_types = [
            "Technical Interview", "Behavioral Interview",
            "HR Interview", "Final Interview",
        ]
        interviewers = []
        for i in range(count):
            gender = random.choice(["Male", "Female"])
            name = self.generate_name()
            interviewers.append({
                "id": i + 1,
                "role_type": "Interviewer",
                "emp_id": self.generate_employee_id("INT"),
                "interview_date": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300))).strftime("%Y-%m-%d"),
                "is_fulltime_interviewer": random.choice(["Yes", "No"]),
                "interview_type": random.choice(interview_types),
                "activity_name": random.choice(self.activity_names),
                "interview_score": round(random.uniform(3.0, 5.0), 1),
                "interview_duration": random.choice(["30 minutes", "45 minutes", "60 minutes", "90 minutes"]),
                "expertise_area": random.choice(self.positions),
                "hr_manager": self.generate_name(),
                "interview_round": random.randint(1, 4),
                "is_weekend": random.choice(["Yes", "No"]),
                "interviewer_level": random.choice(self.job_levels),
                "name": name,
                "sex": gender,
                "organization_name": random.choice(self.companies),
                "job_level_desc": random.choice(self.job_levels),
                "dept_descr0": random.choice(self.departments),
                "dept_descr1": random.choice(self.departments),
                "dept_descr2": random.choice(self.departments),
                "dept_descr3": random.choice(self.departments),
                "hr_status": random.choice(["Active", "Inactive"]),
            })
        return interviewers

    def generate_candidates(self, count: int = 500) -> List[Dict]:
        """生成候选人数据。"""
        education_levels = ["Bachelor", "Master", "PhD", "Associate"]
        interview_results = ["Pass", "Fail", "Pending"]
        majors = [
            "Computer Science", "Software Engineering", "Information Systems",
            "Electrical Engineering", "Business Administration",
        ]
        schools = [
            "MIT", "Stanford University", "UC Berkeley",
            "Carnegie Mellon", "Harvard University",
        ]
        candidates = []
        for i in range(count):
            gender = random.choice(["Male", "Female"])
            name = self.generate_name()
            candidates.append({
                "id": i + 1,
                "activity_name": random.choice(self.activity_names),
                "position_applied": random.choice(self.positions),
                "recruitment_start_date": (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 300))).strftime("%Y-%m-%d"),
                "candidate_name": name,
                "candidate_id": self.generate_employee_id("CAN"),
                "sex": gender,
                "remark": random.choice(["Excellent", "Good", "Average", "Needs Improvement", ""]),
                "work_email": self.generate_email(name),
                "applied_job_level": random.choice(self.job_levels),
                "current_company": random.choice(self.companies),
                "dept_descr0": random.choice(self.departments),
                "dept_descr1": random.choice(self.departments),
                "dept_descr2": random.choice(self.departments),
                "dept_descr3": random.choice(self.departments),
                "education_level": random.choice(education_levels),
                "major": random.choice(majors),
                "graduation_school": random.choice(schools),
                "work_city": random.choice(self.cities),
                "work_experience_years": random.randint(0, 15),
                "current_salary": random.randint(80000, 200000),
                "expected_salary": random.randint(90000, 250000),
                "target_organization_name": random.choice(self.companies),
                "current_job_level_desc": random.choice(self.job_levels),
                "current_dept_name": random.choice(self.departments),
                "current_position": random.choice(self.positions),
                "skill_keywords": random.choice([
                    "Java,Spring,MySQL", "Python,Django,Redis", "React,Vue,JavaScript",
                ]),
                "interview_result": random.choice(interview_results),
                "offer_status": random.choice(["Offer Extended", "No Offer", "Offer Declined", "Offer Accepted"]),
                "onboard_status": random.choice(["Onboarded", "Not Onboarded", "Declined Onboarding"]),
                "referrer_name": random.choice([self.generate_name(), ""]),
                "hr_status": random.choice(["Candidate", "Hired", "Rejected"]),
            })
        return candidates

    def business_tables(
        self,
        activities_count: int = 50,
        interviewers_count: int = 100,
        candidates_count: int = 500,
    ) -> Dict[str, List[Dict]]:
        """返回「表名 -> 行列表」的业务表字典（表名即 SQLite 中的表名）。"""
        return {
            "recruitment_activity_info": self.generate_recruitment_activities(activities_count),
            "recruitment_interviewer_info": self.generate_interviewers(interviewers_count),
            "recruitment_candidate_info": self.generate_candidates(candidates_count),
        }

    # ------------------------------------------------------------------ #
    # 向量元数据（键与 collections_config.json 中的集合名一致）
    # ------------------------------------------------------------------ #
    def generate_table_descriptions(self) -> List[Dict]:
        """生成表描述数据（字段：table_name / description / additional_info）。"""
        return [
            {
                "table_name": "recruitment_interviewer_info",
                "description": "Recruitment interviewer table",
                "additional_info": "Stores all interviewers and their interview records. Structure notes: (1) one interviewer (emp_id) may appear in multiple rows, one per interview; (2) one activity (activity_name) may have multiple interviewers; (3) join to other tables via emp_id and activity_name; (4) use activity_name to distinguish specific recruitment activities.",
            },
            {
                "table_name": "recruitment_activity_info",
                "description": "Recruitment activity list",
                "additional_info": "Used to query recruitment activity progress, applicant numbers and success-rate statistics. Structure notes: (1) one row per activity instance; (2) activities with the same activity_name may run at different times, distinguished by recruitment_start_date; (3) contains aggregate stats (received_resumes, onboard_count, success_rate) suitable for recruitment effectiveness analysis.",
            },
            {
                "table_name": "recruitment_candidate_info",
                "description": "Candidate information table",
                "additional_info": "Records all candidates in recruitment, including the activity they applied to. Structure notes: (1) one candidate (candidate_id) may appear in multiple rows, one per activity applied; (2) one activity (activity_name) may have multiple candidates.",
            },
        ]

    def generate_term_descriptions(self) -> List[Dict]:
        """生成术语映射数据（字段：original_term / standard_name / additional_info）。"""
        return [
            {
                "original_term": "Jim",
                "standard_name": "James Smith",
                "additional_info": "Nickname for the employee `James Smith`.",
            },
            {
                "original_term": "Spring Hiring",
                "standard_name": "Spring Campus Recruitment",
                "additional_info": "Abbreviation for the company's spring campus recruitment activity.",
            },
            {
                "original_term": "Fall Hiring",
                "standard_name": "Fall General Recruitment",
                "additional_info": "Abbreviation for the company's fall general recruitment activity.",
            },
            {
                "original_term": "Big Tech",
                "standard_name": "TechCorp InnovateLab DataFlow",
                "additional_info": "Shorthand referring to the companies TechCorp, InnovateLab and DataFlow.",
            },
            {
                "original_term": "Johnny",
                "standard_name": "John Johnson",
                "additional_info": "Nickname for the employee `John Johnson`.",
            },
            {
                "original_term": "Senior+",
                "standard_name": "",
                "additional_info": "Refers to Senior level and above: Senior, Staff, Principal, Manager, Senior Manager, Director.",
            },
            {
                "original_term": "Tech Dept",
                "standard_name": "Engineering",
                "additional_info": "Generally refers to the department named 'Engineering'.",
            },
        ]

    def generate_query_examples(self) -> List[Dict]:
        """生成 SQL 查询示例（字段：query_text / query_sql，SQLite 方言）。"""
        return [
            {
                "query_text": "List candidates from the most recent Spring Campus Recruitment",
                "query_sql": (
                    "SELECT candidate_name, candidate_id, activity_name, "
                    "target_organization_name, recruitment_start_date "
                    "FROM recruitment_candidate_info "
                    "WHERE activity_name = 'Spring Campus Recruitment' "
                    "AND recruitment_start_date = ("
                    "SELECT MAX(recruitment_start_date) FROM recruitment_candidate_info "
                    "WHERE activity_name = 'Spring Campus Recruitment');"
                ),
            },
            {
                "query_text": "Show all recruitment activities where James Smith served as interviewer",
                "query_sql": (
                    "SELECT DISTINCT activity_name, interview_date "
                    "FROM recruitment_interviewer_info WHERE name = 'James Smith';"
                ),
            },
            {
                "query_text": "Show all recruitment activities that John Johnson participated in",
                "query_sql": (
                    "SELECT DISTINCT activity_name, position_applied, "
                    "recruitment_start_date, target_organization_name "
                    "FROM recruitment_candidate_info "
                    "WHERE candidate_name = 'John Johnson' "
                    "ORDER BY recruitment_start_date DESC;"
                ),
            },
            {
                "query_text": "Compute the recruitment success rate for the Engineering department",
                "query_sql": (
                    "SELECT AVG(success_rate) AS avg_success_rate, "
                    "COUNT(*) AS total_activities "
                    "FROM recruitment_activity_info WHERE department = 'Engineering';"
                ),
            },
            {
                "query_text": "List all candidates who applied for the Java Developer position",
                "query_sql": (
                    "SELECT candidate_name, education_level, work_experience_years, "
                    "current_salary, interview_result "
                    "FROM recruitment_candidate_info "
                    "WHERE position_applied = 'Java Developer';"
                ),
            },
        ]

    def vector_metadata(self) -> Dict[str, List[Dict]]:
        """返回「集合名 -> 行列表」的向量元数据字典。"""
        return {
            "table_descriptions": self.generate_table_descriptions(),
            "term_descriptions": self.generate_term_descriptions(),
            "query_examples": self.generate_query_examples(),
        }

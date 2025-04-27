import csv
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(project_root)

from backend_demo.data_processing.table_operation.table_operations import *


def get_tools_description(tools):
    """获取工具函数的描述。"""
    descriptions = []
    for tool in tools:
        full_description = tool.description
        # 提取 Use cases 及之前的内容
        description_parts = full_description.split("Use cases:")
        description = description_parts[0].strip()
        if len(description_parts) > 1:
            description += "\n\nUse cases:" + description_parts[1].split("\n\n")[0]

        descriptions.append(
            {
                "tool_name": tool.name,
                "description": description,
                "full_description": full_description,
                "args": str(tool.args),
            }
        )
    return descriptions


def generate_csv(tools, output_file):
    """生成包含工具描述的CSV文件。"""
    descriptions = get_tools_description(tools)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["tool_name", "description", "full_description", "args"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for desc in descriptions:
            writer.writerow(desc)


if __name__ == "__main__":
    tools = [
        tool for tool in globals().values() if callable(tool) and hasattr(tool, "name")
    ]

    output_file = os.path.join("data", "datasets", "tools_description.csv")
    generate_csv(tools, output_file)
    print(f"CSV file '{output_file}' has been generated successfully.")

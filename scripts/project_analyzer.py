import os
import sys
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit


def analyze_project(directory):
    total_loc = 0
    total_sloc = 0
    total_comments = 0
    total_multi = 0
    total_blank = 0
    total_files = 0
    max_complexity = 0
    total_complexity = 0
    complexity_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}

    # 定义需要分析的目录和文件类型
    ANALYZE_DIRS = ['backend', 'frontend']
    EXCLUDE_PATTERNS = [
        'node_modules', 'venv', '.git', '__pycache__', 
        'migrations', 'tests', 'test', 'dist', 'build',
        '.next', '.nuxt', 'coverage'
    ]
    
    def analyze_python_file(content):
        """分析 Python 文件的复杂度和指标"""
        analysis = analyze(content)
        complexity = cc_visit(content)
        return analysis, complexity

    def analyze_typescript_file(content):
        """分析 TypeScript 文件的基础指标（不包括复杂度）"""
        # 简单的行数统计
        lines = content.splitlines()
        loc = len(lines)
        blank = len([l for l in lines if not l.strip()])
        comments = len([l for l in lines if l.strip().startswith('//')])
        multi = len([l for l in lines if l.strip().startswith('/*') or l.strip().startswith('*')])
        sloc = loc - blank - comments - multi
        
        return type('Analysis', (), {
            'loc': loc,
            'sloc': sloc,
            'comments': comments,
            'multi': multi,
            'blank': blank
        })

    for root, _, files in os.walk(directory):
        # 检查是否在目标目录中
        relative_path = os.path.relpath(root, directory)
        base_dir = relative_path.split(os.path.sep)[0]
        if base_dir not in ANALYZE_DIRS:
            continue

        # 检查是否在排除目录中
        if any(pattern in root.split(os.path.sep) for pattern in EXCLUDE_PATTERNS):
            continue
            
        for file in files:
            is_python = file.endswith('.py')
            is_typescript = file.endswith(('.ts', '.tsx')) and not file.endswith(('.d.ts', '.test.ts', '.test.tsx', '.spec.ts', '.spec.tsx'))
            
            if not (is_python or is_typescript):
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if is_python:
                    # Python 文件完整分析
                    analysis, complexity = analyze_python_file(content)
                    total_loc += analysis.loc
                    total_sloc += analysis.sloc
                    total_comments += analysis.comments
                    total_multi += analysis.multi
                    total_blank += analysis.blank
                    
                    if complexity:
                        file_max_complexity = max(item.complexity for item in complexity)
                        max_complexity = max(max_complexity, file_max_complexity)
                        total_complexity += sum(item.complexity for item in complexity)

                        for item in complexity:
                            if item.complexity <= 5:
                                complexity_counts["A"] += 1
                            elif item.complexity <= 10:
                                complexity_counts["B"] += 1
                            elif item.complexity <= 20:
                                complexity_counts["C"] += 1
                            elif item.complexity <= 30:
                                complexity_counts["D"] += 1
                            elif item.complexity <= 40:
                                complexity_counts["E"] += 1
                            else:
                                complexity_counts["F"] += 1

                else:
                    # TypeScript 文件只分析基础指标
                    analysis = analyze_typescript_file(content)
                    total_loc += analysis.loc
                    total_sloc += analysis.sloc
                    total_comments += analysis.comments
                    total_multi += analysis.multi
                    total_blank += analysis.blank

                total_files += 1

            except Exception as e:
                print(f"警告: 处理文件时出错 {file_path}: {str(e)}")
                continue

    return {
        "total_files": total_files,
        "total_loc": total_loc,
        "total_sloc": total_sloc,
        "total_comments": total_comments,
        "total_multi": total_multi,
        "total_blank": total_blank,
        "max_complexity": max_complexity,
        "avg_complexity": total_complexity / total_files if total_files > 0 else 0,
        "complexity_counts": complexity_counts,
    }


def main():
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("正在分析项目...")
    results = analyze_project(project_root)

    print("\n项目分析结果 (仅包含 backend 和 frontend 目录):")
    print("注意: TypeScript 文件仅统计行数，不包含复杂度分析")
    print(f"总文件数: {results['total_files']}")
    print(f"总行数 (LOC): {results['total_loc']}")
    print(f"代码行数 (SLOC): {results['total_sloc']}")
    print(f"注释行数: {results['total_comments']}")
    print(f"多行注释行数: {results['total_multi']}")
    print(f"空白行数: {results['total_blank']}")
    print(f"\n最大圈复杂度: {results['max_complexity']}")
    print(f"平均圈复杂度: {results['avg_complexity']:.2f}")

    print("\n圈复杂度分布:")
    for grade, count in results["complexity_counts"].items():
        print(f"  等级 {grade}: {count}")


if __name__ == "__main__":
    main()

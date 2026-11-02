import json
import time
import subprocess
import re
import csv
import os
from openai import OpenAI

# ================= 1. 配置你的大模型 API =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_env():
    """读取脚本同目录下的 .env 文件（KEY=VALUE），无需安装任何第三方库。"""
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_env()

API_KEY = os.getenv("DEEPSEEK_API_KEY")  # key 存在 .env 文件里，不写进代码
if not API_KEY:
    raise SystemExit(
        "❌ 未找到 DeepSeek 的 API Key。\n"
        f"   请在文件 {os.path.join(SCRIPT_DIR, '.env')} 里写一行：\n"
        '   DEEPSEEK_API_KEY="sk-你的key"'
    )
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-coder"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(prompt):
    """调用大模型，带简单的重试机制防止网络波动"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  [警告] API 调用失败，正在重试 ({attempt+1}/3)... 错误: {e}")
            time.sleep(2)
    return ""

def extract_code(text):
    """从大模型的回答中提取 python 代码块"""
    if not text: return ""
    # 匹配 ```python 和 ``` 之间的内容
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match: return match.group(1)
    return text

# ================= 2. 核心运行与计时模块 =================
def run_code_and_measure(full_code, task_id, run_times=1):
    """把代码写入临时文件，运行并计时"""
    # 清理非法字符作为文件名
    safe_name = task_id.replace('/', '_')
    temp_file = f"temp_{safe_name}.py"
    
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    total_time = 0
    passed = False
    error_msg = ""
    
    for _ in range(run_times):
        start_time = time.perf_counter()
        try:
            # 限制超时 10 秒，防止死循环卡死你的电脑
            result = subprocess.run(["python", temp_file], capture_output=True, text=True, timeout=10)
            end_time = time.perf_counter()
            
            if result.returncode == 0:
                passed = True
                total_time += (end_time - start_time)
            else:
                passed = False
                error_msg = result.stderr
                break # 如果报错了，就不用重复跑了
        except subprocess.TimeoutExpired:
            passed = False
            error_msg = "Timeout (超过10秒)"
            break
            
    # 清理临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    avg_time = total_time / run_times if passed else 0
    return passed, error_msg, avg_time

# ================= 3. 主实验流程 =================
def main():
    dataset_file = os.path.join(SCRIPT_DIR, "HumanEval.jsonl", "human-eval-v2-20210705.jsonl")
    csv_file = os.path.join(SCRIPT_DIR, "experiment_results.csv")
    
    # 收集所有结果，最后一次性写入，防止文件读写冲突
    results_data = [] 
    
    # 你可以在这里修改要跑的题目数量
    tasks_to_run = 10 
    count = 0

    print("开始读取数据集并运行实验...")
    
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if count >= tasks_to_run: break
            
            data = json.loads(line)
            task_id = data["task_id"]
            prompt = data["prompt"]
            test_code = data["test"]
            entry_point = data["entry_point"]
            count += 1
            
            print(f"\n[{count}/{tasks_to_run}] 正在处理题目: {task_id}")

            # -------- 阶段一：初始生成与正确性验证 --------
            base_prompt = f"{prompt}\n# 请补全上述代码，直接输出代码即可"
            raw_response = call_llm(base_prompt)
            code = extract_code(raw_response)
            
            if not code.strip():
                print("  ❌ 模型返回为空或无法提取代码，跳过。")
                results_data.append([task_id, False, 0, False, 0, 0])
                continue

            # 拼装：题目 + 代码 + 测试 + 执行测试
            full_code = f"{prompt}\n{code}\n\n{test_code}\n\ncheck({entry_point})"
            passed, err, base_time = run_code_and_measure(full_code, task_id)
            
            if not passed:
                # 尝试修正一次
                print("  ⚠️ 初次生成未通过测试，尝试让模型修正...")
                fix_prompt = f"代码运行报错：\n{err}\n请分析原因并修改代码，确保通过测试。\n题目：{prompt}\n有问题的代码：\n{code}"
                fixed_code = extract_code(call_llm(fix_prompt))
                full_code = f"{prompt}\n{fixed_code}\n\n{test_code}\n\ncheck({entry_point})"
                passed, err, base_time = run_code_and_measure(full_code, task_id)
                code = fixed_code # 更新当前代码为修正后的代码
                
                if not passed:
                    print("  ❌ 模型无法修正代码，放弃该题。")
                    results_data.append([task_id, False, 0, False, 0, 0])
                    continue
            
            print(f"  ✅ 初始正确！耗时: {base_time:.5f}s")

            # -------- 阶段二：PerfCodeGen 优化性能 --------
            print("  正在尝试性能优化...")
            perf_prompt = f"""
            Good job! You generated the correct solution. Now let's optimize the time performance.
            Here is the current code:
            {code}
            
            Please refine it so that it consumes less time. 
            Make sure your refined solution is functionally equivalent. 
            Output the solution wrapped in ```python```.
            """
            opt_code = extract_code(call_llm(perf_prompt))
            
            if not opt_code.strip():
                print("  ❌ 模型优化返回为空，保留原代码。")
                results_data.append([task_id, True, round(base_time, 5), False, 0, 1.0])
                continue

            opt_full_code = f"{prompt}\n{opt_code}\n\n{test_code}\n\ncheck({entry_point})"
            
            # 论文要求：优化后的代码必须重复跑多次（这里跑 12 次）去除噪声
            print("  正在验证优化后的代码 (12次重复计时)...")
            opt_passed, opt_err, opt_time = run_code_and_measure(opt_full_code, task_id, run_times=12)
            
            if not opt_passed:
                print(f"  ❌ 优化失败 (代码报错或超时)，回退到原代码。")
                results_data.append([task_id, True, round(base_time, 5), False, 0, 1.0])
            else:
                speedup = base_time / opt_time if opt_time > 0 else 0
                print(f"  🚀 优化成功！优化后耗时: {opt_time:.5f}s, 加速比: {speedup:.2f}x")
                results_data.append([task_id, True, round(base_time, 5), True, round(opt_time, 5), round(speedup, 2)])

    # ================= 4. 把数据统一保存到 CSV =================
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Task_ID", "Baseline_Passed", "Baseline_Time", "Optimized_Passed", "Optimized_Time", "Speedup"])
        writer.writerows(results_data)
        
    print(f"\n🎉 实验完毕！数据已成功保存到 {csv_file} 文件中。")
    print("请打开 WPS/Excel 查看表格，准备开始写报告！")

if __name__ == "__main__":
    main()
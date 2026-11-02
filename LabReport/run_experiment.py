import json
import time
import subprocess
import re
import csv
import os
from openai import OpenAI

# ================= 1. 配置你的大模型 API =================
API_KEY = "sk-d8d5ade412094376996fe33b72f5e6c4"  # 必须改这里！
BASE_URL = "https://api.deepseek.com" # 如果是通义千问或GPT，换这里的URL
MODEL_NAME = "deepseek-coder" # 如果是 GPT 换 "gpt-4o-mini"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""

def extract_code(text):
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match: return match.group(1)
    return text

# ================= 2. 核心运行与计时模块 =================
def run_code_and_measure(full_code, file_name, run_times=1):
    temp_file = f"{file_name}_temp.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(full_code)
    
    total_time = 0
    passed = False
    error_msg = ""
    
    for _ in range(run_times):
        start_time = time.perf_counter()
        try:
            # 限制超时 10 秒，防止死循环卡死
            result = subprocess.run(["python", temp_file], capture_output=True, text=True, timeout=10)
            end_time = time.perf_counter()
            if result.returncode == 0:
                passed = True
                total_time += (end_time - start_time)
            else:
                passed = False
                error_msg = result.stderr
                break
        except subprocess.TimeoutExpired:
            passed = False
            error_msg = "Timeout (超过10秒)"
            break
            
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    avg_time = total_time / run_times if passed else 0
    return passed, error_msg, avg_time

# ================= 3. 主实验流程 =================
def main():
    dataset_file = "HumanEval.jsonl"
    # 结果保存到 CSV
    csv_file = "experiment_results.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Task_ID", "Baseline_Passed", "Baseline_Time", "Optimized_Passed", "Optimized_Time", "Speedup"])

    # 我们只跑前 10 道题作为演示，你可以改成 30 或 50
    tasks_to_run = 10
    count = 0

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

            # --- 阶段一：初始生成与正确性验证 ---
            base_prompt = f"{prompt}\n# 请补全上述代码"
            code = extract_code(call_llm(base_prompt))
            
            # 拼装：题目 + 代码 + 测试 + 执行测试
            full_code = f"{prompt}\n{code}\n\n{test_code}\n\ncheck({entry_point})"
            passed, err, base_time = run_code_and_measure(full_code, task_id.replace('/', '_'))
            
            if not passed:
                print(f"  ❌ 初始生成未通过测试 (可能是题目太难或代码有bug)")
                writer.writerow([task_id, False, 0, False, 0, 0])
                continue
                
            print(f"  ✅ 初始正确！耗时: {base_time:.5f}s")

            # --- 阶段二：PerfCodeGen 优化性能 ---
            perf_prompt = f"""
            Good job! You generated the correct solution. Now let's optimize the time performance.
            Here is the current code:
            {code}
            
            Please refine it so that it consumes less time. 
            Make sure your refined solution is functionally equivalent. 
            Output the solution wrapped in ```python```.
            """
            opt_code = extract_code(call_llm(perf_prompt))
            opt_full_code = f"{prompt}\n{opt_code}\n\n{test_code}\n\ncheck({entry_point})"
            
            # 论文要求：优化后的代码必须重复跑多次（这里跑 12 次）去除噪声
            print("  正在验证优化后的代码 (12次重复计时)...")
            opt_passed, opt_err, opt_time = run_code_and_measure(opt_full_code, task_id.replace('/', '_'), run_times=12)
            
            if not opt_passed:
                print(f"  ❌ 优化失败 (代码报错或超时)，回退到原代码。")
                writer.writerow([task_id, True, round(base_time, 5), False, 0, 1.0])
            else:
                speedup = base_time / opt_time if opt_time > 0 else 0
                print(f"  🚀 优化成功！优化后耗时: {opt_time:.5f}s, 加速比: {speedup:.2f}x")
                writer.writerow([task_id, True, round(base_time, 5), True, round(opt_time, 5), round(speedup, 2)])

    print(f"\n🎉 实验完毕！数据已保存到 {csv_file}，你可以直接用 Excel 打开写报告了。")

if __name__ == "__main__":
    main()
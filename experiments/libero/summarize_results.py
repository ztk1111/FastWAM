"""
LIBERO 评估结果汇总模块。

该模块负责解析和汇总 LIBERO 多 GPU 评估生成的 JSON 结果文件，
生成以下输出：
1. summary.json: 包含所有套件的详细统计信息和每任务结果。
2. summary.csv: 按套件汇总的成功率/耗时表格（转置格式，适合报告）。
3. task_success_rates.csv: 每任务的详细成功率表格。

输出表格示例:
    summary.csv (转置):
        Task Suite          libero_spatial  libero_object  Overall
        Success Rate (%)    85.00          78.00          81.50
        Average Time (s)    120.50         95.30          107.90

    task_success_rates.csv:
        Task                       Description      Success Rate (%)
        libero_spatial_0           pick up foam     80.00
        libero_spatial_1           open drawer      90.00
"""

import os
import json
import argparse
from collections import defaultdict
import pandas as pd
import math

def format_time(seconds):
    """将秒数格式化为人类可读的时长字符串。

    格式规则:
        - 小于 1 分钟: "SSs" (如 "45s")
        - 小于 1 小时: "MMmSSs" (如 "05m30s")
        - 1 小时以上: "HHhMMmSSs" (如 "01h15m30s")

    Args:
        seconds: 秒数（浮点数或整数）。

    Returns:
        str: 格式化后的时长字符串。

    示例:
        >>> format_time(45)
        '45s'
        >>> format_time(90)
        '01m30s'
        >>> format_time(3661)
        '01h01m01s'
    """
    seconds = round(seconds)  # 四舍五入到整数秒

    if seconds < 60:
        return f"{seconds:02d}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"
    else:
        hours = seconds // 3600
        remaining = seconds % 3600
        minutes = remaining // 60
        remaining_seconds = remaining % 60
        return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"

def summarize_results(output_dir):
    """汇总所有评估结果，生成 CSV 和 JSON 摘要。

    遍历 output_dir 下所有套件目录（libero_spatial/object/goal/10/90），
    读取每个 gpu<ID>_task<ID>_results.json 文件，统计成功率、耗时和 PSNR。

    Args:
        output_dir: 包含评估结果文件的根目录。目录结构应为:
            <output_dir>/
            ├── libero_spatial/
            │   ├── gpu0_task0_results.json
            │   ├── gpu1_task1_results.json
            │   └── videos/
            ├── libero_object/
            │   └── ...
            ├── summary.csv
            ├── task_success_rates.csv
            └── summary.json
    """
    # 为每个套件存储统计信息
    suite_stats = defaultdict(lambda: {
        'total_tasks': 0,
        'total_trials': 0,
        'total_successes': 0,
        'total_time': 0,
        'max_time': 0,
        'psnr_sum': 0.0,
        'psnr_count': 0
    })
    
    # 存储详细的每任务结果
    task_results = {}
    has_psnr_metric = False  # 标记是否有 PSNR 指标可用

    # 遍历所有套件目录
    for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
        suite_dir = os.path.join(output_dir, suite)
        if not os.path.exists(suite_dir):
            continue  # 该套件未评估，跳过

        # 读取该套件下所有结果文件
        for filename in os.listdir(suite_dir):
            if not filename.startswith('gpu') or not filename.endswith('_results.json'):
                continue

            with open(os.path.join(suite_dir, filename), 'r') as f:
                result = json.load(f)

            # 从文件名提取任务 ID: "gpu0_task5_results.json" -> 5
            parts = filename.split('_')
            task_id = int(parts[1].replace('task', ''))

            # 创建任务唯一标识: "libero_spatial_5"
            task_key = f"{suite}_{task_id}"

            stats = suite_stats[suite]
            stats['total_tasks'] += 1
            stats['total_trials'] += result['total_episodes']
            stats['total_successes'] += result['successes']
            stats['total_time'] += result['duration']
            stats['max_time'] = max(stats['max_time'], result['duration'])
            if 'future_video_psnr_mean' in result:
                has_psnr_metric = True
                if result['future_video_psnr_mean'] is not None:
                    stats['psnr_sum'] += float(result['future_video_psnr_mean'])
                    stats['psnr_count'] += 1
            
            # 存储每任务的详细结果
            task_result = {
                'success_rate': result['successes'] / result['total_episodes'] * 100,
                'duration': result['duration'],
                'total_episodes': result['total_episodes'],
                'successes': result['successes'],
                'task_description': result['task_description'] if 'task_description' in result else ''
            }
            if 'future_video_psnr_mean' in result:
                task_result['future_video_psnr_mean'] = (
                    float(result['future_video_psnr_mean'])
                    if result['future_video_psnr_mean'] is not None
                    else None
                )
            task_results[task_key] = task_result
    
    # Print summary results
    print("\n=== Evaluation Results Summary ===")
    print("\nStatistics for each task suite:")
    
    total_success_rate = 0
    total_time = 0
    total_suites = 0
    overall_psnr_sum = 0.0
    overall_psnr_count = 0
    
    # Prepare DataFrame rows
    df_data = {
        'Task Suite': [],
        'Success Rate (%)': [],
        'Average Time (s)': [],
        'Max Time (s)': []
    }
    if has_psnr_metric:
        df_data['Average Future PSNR (dB)'] = []
    
    for suite, stats in suite_stats.items():
        if stats['total_trials'] > 0:
            success_rate = stats['total_successes'] / stats['total_trials'] * 100
            avg_time = stats['total_time'] / stats['total_tasks']
            max_time = stats['max_time']
            suite_avg_psnr = None
            if has_psnr_metric:
                suite_avg_psnr = (
                    stats['psnr_sum'] / stats['psnr_count']
                    if stats['psnr_count'] > 0
                    else None
                )
            
            print(f"\n{suite}:")
            print(f"- Tasks completed: {stats['total_tasks']}")
            print(f"- Total attempts: {stats['total_trials']}")
            print(f"- Successful attempts: {stats['total_successes']}")
            print(f"- Success rate: {success_rate:.2f}%")
            print(f"- Total time: {format_time(stats['total_time'])}")
            print(f"- Average time per task: {format_time(avg_time)}")
            print(f"- Longest task time: {format_time(max_time)}")
            if has_psnr_metric:
                if suite_avg_psnr is not None:
                    print(f"- Average future-video PSNR: {suite_avg_psnr:.4f} dB")
                else:
                    print("- Average future-video PSNR: N/A")
            
            # Append to DataFrame rows
            df_data['Task Suite'].append(suite)
            df_data['Success Rate (%)'].append(f"{success_rate:.2f}")
            df_data['Average Time (s)'].append(f"{avg_time:.2f}")
            df_data['Max Time (s)'].append(f"{max_time:.2f}")
            if has_psnr_metric:
                df_data['Average Future PSNR (dB)'].append(
                    f"{suite_avg_psnr:.4f}" if suite_avg_psnr is not None else "N/A"
                )
            
            total_success_rate += success_rate
            total_time += stats['total_time']
            total_suites += 1
            if has_psnr_metric:
                overall_psnr_sum += stats['psnr_sum']
                overall_psnr_count += stats['psnr_count']
    
    if total_suites > 0:
        print("\nOverall statistics:")
        avg_success_rate = total_success_rate/total_suites
        avg_task_time = total_time/sum(s['total_tasks'] for s in suite_stats.values())
        max_task_time = max(s['max_time'] for s in suite_stats.values())
        overall_avg_psnr = None
        if has_psnr_metric:
            overall_avg_psnr = overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        
        print(f"- Average success rate: {avg_success_rate:.2f}%")
        print(f"- Total time: {format_time(total_time)}")
        print(f"- Average time per task: {format_time(avg_task_time)}")
        print(f"- Longest task time: {format_time(max_task_time)}")
        if has_psnr_metric:
            if overall_avg_psnr is not None:
                print(f"- Average future-video PSNR: {overall_avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")
        
        # 添加总体汇总行
        df_data['Task Suite'].append('Overall')
        df_data['Success Rate (%)'].append(f"{avg_success_rate:.2f}")
        df_data['Average Time (s)'].append(f"{avg_task_time:.2f}")
        df_data['Max Time (s)'].append(f"{max_task_time:.2f}")
        if has_psnr_metric:
            df_data['Average Future PSNR (dB)'].append(
                f"{overall_avg_psnr:.4f}" if overall_avg_psnr is not None else "N/A"
            )

    # 创建并保存 DataFrame 为 summary.csv（转置格式）
    df = pd.DataFrame(df_data)

    # 使用检查点路径的最后一部分作为 CSV 标题
    ckpt_path = os.environ.get('CKPT', '')
    title = os.path.basename(ckpt_path) if ckpt_path else 'Results'

    # 转置 DataFrame，使 Task Suite 作为列名
    df = df.set_index('Task Suite').T

    # 写入 CSV（第一行为标题），
    with open(os.path.join(output_dir, 'summary.csv'), 'w') as f:
        f.write(f"{title}\n")  # 写入标题行
        df.to_csv(f)

    # 准备每任务成功率数据
    task_success_data = {
        'Task': [],
        'Description': [],
        'Success Rate (%)': []
    }
    if has_psnr_metric:
        task_success_data['Future Video PSNR (dB)'] = []

    # 按套件分组任务
    suite_tasks = defaultdict(list)
    for task in task_results:
        suite = task.split('_')[0] + '_' + task.split('_')[1]
        suite_tasks[suite].append(task)

    # 对每个套件内的任务按 ID 排序
    for suite in suite_tasks:
        suite_tasks[suite].sort(key=lambda x: int(x.split('_')[-1]))

    # 填充每任务成功率表格
    for suite in sorted(suite_tasks.keys()):
        for task in suite_tasks[suite]:
            result = task_results[task]
            task_success_data['Task'].append(task)
            task_success_data['Description'].append(
                result['task_description'] if 'task_description' in result else ''
            )
            task_success_data['Success Rate (%)'].append(f"{result['success_rate']:.2f}")
            if has_psnr_metric:
                psnr = result['future_video_psnr_mean'] if 'future_video_psnr_mean' in result else None
                task_success_data['Future Video PSNR (dB)'].append(
                    f"{psnr:.4f}" if psnr is not None else "N/A"
                )

    suite_stats_output = {}
    for suite, stats in suite_stats.items():
        suite_stats_output[suite] = {
            'total_tasks': stats['total_tasks'],
            'total_trials': stats['total_trials'],
            'total_successes': stats['total_successes'],
            'total_time': stats['total_time'],
            'max_time': stats['max_time'],
        }
        if has_psnr_metric:
            suite_stats_output[suite]['average_future_video_psnr'] = (
                stats['psnr_sum'] / stats['psnr_count'] if stats['psnr_count'] > 0 else None
            )
    
    # 创建并保存每任务成功率 CSV
    task_success_df = pd.DataFrame(task_success_data)
    task_success_df.to_csv(os.path.join(output_dir, 'task_success_rates.csv'), index=False)

    # 保存详细的 JSON 汇总
    summary_file = os.path.join(output_dir, 'summary.json')
    overall_stats = {
        'average_success_rate': total_success_rate/total_suites if total_suites > 0 else 0,
        'total_time': total_time,
        'average_task_time': total_time/sum(s['total_tasks'] for s in suite_stats.values()) if suite_stats else 0,
    }
    if has_psnr_metric:
        overall_stats['average_future_video_psnr'] = (
            overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        )

    with open(summary_file, 'w') as f:
        json.dump({
            'run_id': os.path.basename(output_dir),
            'ckpt': os.environ.get('CKPT', ''),  # 检查点路径
            'config': os.environ.get('CONFIG', ''),  # 配置路径
            'suite_stats': suite_stats_output,
            'task_results': task_results,
            'overall': overall_stats
        }, f, indent=4)

    print(f"\n=== Run Information ===")
    print(f"Run ID: {os.path.basename(output_dir)}")
    print(f"Results directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary CSV: {os.path.join(output_dir, 'summary.csv')}")
    print(f"Task success rates CSV: {os.path.join(output_dir, 'task_success_rates.csv')}")

    # 打印每任务成功率表格
    print("\n=== Task Success Rates ===")
    print(task_success_df.to_string(index=False))

    # 打印转置后的汇总表
    print("\n=== Results Table ===")
    print(df.to_string(index=False))


def main():
    """命令行入口：解析 --output_dir 参数并执行结果汇总。

    使用示例:
        python experiments/libero/summarize_results.py --output_dir /path/to/eval_results
    """
    parser = argparse.ArgumentParser(description="LIBERO 评估结果汇总工具")
    parser.add_argument('--output_dir', type=str, required=True,
                      help='包含评估结果文件的根目录')
    args = parser.parse_args()

    summarize_results(args.output_dir)

if __name__ == '__main__':
    main()

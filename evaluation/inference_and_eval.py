"""
ProactiveMobile — inference + evaluation (single file, local vLLM backend).

Runs local in-process vLLM inference and scores the ProactiveMobile benchmark.
Prompt construction and all metrics (Type-Acc / Precision / Recall / F1 / FTR) are
kept consistent with the evaluation protocol used in the paper, so numbers stay
comparable. Type-Acc uses deterministic sequence matching (no set() dedup), so
results are fully reproducible across runs.

Usage:
  # 1) inference with local vLLM (TP*PP = #GPUs; Qwen2.5-VL-7B has 28 heads -> TP in 1/2/4/7/14)
  python evaluation/inference_and_eval.py --mode eval \
      --model_path <model-path> \
      --eval_file benchmark_zh.json --function_pool_file function_pool.json \
      --output_format thinking recommendation \
      --tensor_parallel_size 4 --pipeline_parallel_size 2 \
      --output_path ./result

  # 2) compute metrics (Type-Acc / Precision / Recall / F1 / FTR, by difficulty x GUI/TEXT)
  python evaluation/inference_and_eval.py --mode compute \
      --model_path <model-path> \
      --eval_file benchmark_zh.json --function_pool_file function_pool.json \
      --output_path ./result
"""
import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import time
from typing import Dict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


parser = argparse.ArgumentParser(description="ProactiveMobile eval (local vLLM backend)")
parser.add_argument("--mode", type=str, default="compute", choices=["eval", "compute"],
                    help="eval = run inference; compute = score saved predictions")
parser.add_argument("--model_path", type=str, required=True,
                    help="model path; its basename is the result key")
parser.add_argument("--eval_file", type=str, default="./benchmark_zh.json")
parser.add_argument("--function_pool_file", type=str, default="./function_pool.json")
parser.add_argument("--output_path", type=str, default="./result")
parser.add_argument("--output_format", nargs="*", choices=["thinking", "recommendation"],
                    default=[], help="prompt/parse format; e.g. --output_format thinking recommendation")
parser.add_argument("--image_root", type=str, default="",
                    help="root dir for images; combined with each image path (see --image_prefix)")
parser.add_argument("--image_prefix", type=str, default="",
                    help="if set, this leading prefix in dataset image paths is replaced by "
                         "--image_root (e.g. --image_prefix Benchmark/ --image_root /data/images). "
                         "If unset, --image_root is joined in front of the path.")
parser.add_argument("--image_max_pixels", type=int, default=1280 * 1280,
                    help="down-sample images above this many pixels (aspect preserved)")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="default 0 = greedy decoding (stable, reproducible). "
                         "The paper used 1.0; greedy gives lower-variance FTR.")
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--max_tokens", type=int, default=2048)
parser.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="tensor-parallel GPUs; must divide the model's attention-head "
                         "count (Qwen2.5-VL-7B has 28 heads -> use 1/2/4/7/14)")
parser.add_argument("--pipeline_parallel_size", type=int, default=1,
                    help="pipeline-parallel stages; TP*PP = total GPUs "
                         "(e.g. TP=4 PP=2 uses 8 GPUs for a 28-head model)")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                    help="fraction of GPU memory vLLM may use")
parser.add_argument("--max_model_len", type=int, default=None,
                    help="max context length; None = model default")


def build_prompt_text(each_data, function_pool, args):
    """构建纯文本prompt（不包含图像占位符，图像以base64通过vLLM接口传入）"""
    if 'thinking' in args.output_format and 'recommendation' in args.output_format:
        user_prompt = f"""
    <Role>
    你是一个专业的手机智能助手评估专家，专门分析手机主动智能推荐系统的决策过程。你需要基于多维度信息预测用户的操作意图，并判断是否应该主动向用户推荐相关功能。

    你的输出分为三个阶段：
    1. <think>：先用自然语言推理并表达你的思考过程。
    2. <rec>：基于思考过程，用自然语言陈述推荐的行动。
    3. <function>：再将上述行动转化为结构化函数调用序列。

    其中，<rec> 应该是 <function> 的结果，且 <rec> 和 <function> 二者必须严格一致 —— <function> 是 <rec> 的精确结构化翻译。
    </Role>

    <Task>
    基于以下四类输入信息进行综合分析：
    - **用户画像信息profile**：用户的个人偏好、使用习惯、历史行为模式
    - **手机环境信息phone**：当前设备状态、网络环境、电量状态等
    - **世界信息world**：时间、地点、天气、日程安排等外部环境
    - **用户行为轨迹trace**：近期的操作序列和行为模式

    你的任务是：
    1. 深度分析用户的真实需求和意图
    2. 评估需求的紧迫性和明确性
    3. 判断是否需要主动推荐
    4. 如需推荐，从给定功能池中选择合适的执行序列

    **重要说明**：
    - 只有当推荐可以通过提供的功能完全实现时，才返回功能序列
    - 如果不需要推荐或无法通过现有功能实现，返回空列表
    - 功能序列应按执行顺序排列
    </Task>

    <Analysis_Framework>
    请按以下框架进行分析：

    1. **需求识别**：从轨迹中识别用户的潜在需求，结合画像、环境、世界信息（特别是时间、地点、日程）进行交叉验证
    2. **紧迫性评估**：判断需求是否需要立即处理 —— 重点考虑当前时间、日程安排、设备状态（如低电量、弱网）是否构成紧迫性触发条件
    3. **功能匹配**：检查需求是否可通过现有功能池中的功能完全满足 —— 功能参数必须可填充且无缺失关键参数
    4. **推荐决策**：基于以上分析决定是否推荐及推荐内容 —— 仅当需求明确、紧迫、且功能可完全执行时才推荐
    </Analysis_Framework>

    <Output_Format>

        输出格式必须严格遵循：
        <think>你的思考过程</think><rec>你的自然语言推理与推荐决策</rec><function>对应的结构化函数调用JSON</function>

    </Output_Format>

    <Output_Requirements>
        **<think> 部分要求：**
        思考过程应包括对用户需求的深度分析，结合给定的多维度信息进行推理。

        **<rec> 部分要求：**
        给出的推荐动作需要是具体、能够在手机上以有限步骤执行。预测的任务不应该是推荐式，而是具体的、在手机上能够执行的一系列动作。

        **<function> 部分要求（JSON格式）：**
        ```json
    {{
        "model_recommendation": [
            {{
                "name":"function_name_1",
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": ""
                }}
            }},
            {{
                "name":"function_name_2",
                "parameters": {{
                    "param1": "value1",
                    "param2": "",
                    "param3": ""
                }}
            }}
        ]
    }}
    ```

    **JSON格式要求**：
    - 所有键和字符串值必须用双引号（""）包裹
    - 键值对之间用逗号分隔，最后一个键值对后不加逗号
    - JSON字符串内部的引号使用单引号（''）
    - 禁止使用注释（// 或 /* */）
    - 每个功能必须包含 "name" 和 "parameters" 字段
    - parameters 中仅包含该功能执行所必需的参数；可选参数若无值，可设为空字符串 "" 或省略该键
    - 如无需推荐，<rec>为'无推荐'，model_recommendation字段为空数组[]

    一致性强制要求：
    - <function> 中的函数序列必须完全对应 <rec> 中描述的操作
    - 若 <rec> 说“不推荐”，但 <function> 有内容 → 输出无效
    - 若 <rec> 有推荐内容，但可用功能列表中找不到对应的功能，那么输出的 <function> 应为空列表

    函数要求：
    - 在我们给的函数列表中，每个函数都有名字、描述、相似函数和参数。  
    - 每个参数有对应的描述、类型和是否必填。如果非必填，则可以省略掉；否则必须填写。
    - 如果一个参数的类型为"dict"，那么"value"中是该参数的具体结构和字段。请你确保每一个字段都要填写到；如果该参数的类型不为"dict"，且"value"不为"无法枚举"，则根据参数类型从"value"中选择一个(参数类型为str, bool时)或多个值(参数类型为list时)进行填写。
    - 不要再新增加任何函数或参数。
    - 请仔细分析指令的完整含义，考虑是否需要多个步骤或函数组合来实现该功能。如果指令需要多个函数组合完成，应按执行顺序列出所有相关函数
    - 有可能该指令并没有办法对应到合适的函数，这时候"function"字段对应空列表[]
    - 对于函数中未使用或未知的参数，保持为空


    </Output_Requirements>

    <Input_Data>
    ### 参考数据：
    {json.dumps(each_data['reference_information'], ensure_ascii=False)}

    ### 可用功能列表：
    {json.dumps(function_pool, ensure_ascii=False)}
    </Input_Data> 

    请仔细分析输入数据（特别是图片数据中的所有可见信息），基于分析框架进行推理，并严格按照JSON格式输出推荐结果。
"""
    elif 'recommendation' in args.output_format:
        user_prompt = f"""
    <Role>
    你是一个专业的手机智能助手评估专家，专门分析手机主动智能推荐系统的决策过程。你需要基于多维度信息预测用户的操作意图，并判断是否应该主动向用户推荐相关功能。

    你的输出分为两个阶段：
    1. <rec>：先用自然语言陈述推荐的行动。
    2. <function>：再将上述行动转化为结构化函数调用序列。

    二者必须严格一致 —— <function> 是 <rec> 的精确结构化翻译。
    </Role>

    <Task>
    基于以下四类输入信息进行综合分析：
    - **用户画像信息profile**：用户的个人偏好、使用习惯、历史行为模式
    - **手机环境信息phone**：当前设备状态、网络环境、电量状态等
    - **世界信息world**：时间、地点、天气、日程安排等外部环境
    - **用户行为轨迹trace**：近期的操作序列和行为模式

    你的任务是：
    1. 深度分析用户的真实需求和意图
    2. 评估需求的紧迫性和明确性
    3. 判断是否需要主动推荐
    4. 如需推荐，从给定功能池中选择合适的执行序列

    **重要说明**：
    - 只有当推荐可以通过提供的功能完全实现时，才返回功能序列
    - 如果不需要推荐或无法通过现有功能实现，返回空列表
    - 功能序列应按执行顺序排列
    </Task>

    <Analysis_Framework>
    请按以下框架进行分析：

    1. **需求识别**：从轨迹中识别用户的潜在需求，结合画像、环境、世界信息（特别是时间、地点、日程）进行交叉验证
    2. **紧迫性评估**：判断需求是否需要立即处理 —— 重点考虑当前时间、日程安排、设备状态（如低电量、弱网）是否构成紧迫性触发条件
    3. **功能匹配**：检查需求是否可通过现有功能池中的功能完全满足 —— 功能参数必须可填充且无缺失关键参数
    4. **推荐决策**：基于以上分析决定是否推荐及推荐内容 —— 仅当需求明确、紧迫、且功能可完全执行时才推荐
    </Analysis_Framework>

    <Output_Format>

        输出格式必须严格遵循：
        <rec>你的自然语言推理与推荐决策</rec><function>对应的结构化函数调用JSON</function>

    </Output_Format>

    <Output_Requirements>
        **<rec> 部分要求：**
        给出的推荐动作需要是具体、能够在手机上以有限步骤执行。预测的任务不应该是推荐式，而是具体的、在手机上能够执行的一系列动作。

        **<function> 部分要求（JSON格式）：**
        ```json
    {{
        "model_recommendation": [
            {{
                "name":"function_name_1",
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": ""
                }}
            }},
            {{
                "name":"function_name_2",
                "parameters": {{
                    "param1": "value1",
                    "param2": "",
                    "param3": ""
                }}
            }}
        ]
    }}
    ```

    **JSON格式要求**：
    - 所有键和字符串值必须用双引号（""）包裹
    - 键值对之间用逗号分隔，最后一个键值对后不加逗号
    - JSON字符串内部的引号使用单引号（''）
    - 禁止使用注释（// 或 /* */）
    - 每个功能必须包含 "name" 和 "parameters" 字段
    - 如无需推荐，<rec>为'无推荐'，model_recommendation字段为空数组[]

    一致性强制要求：
    - <function> 中的函数序列必须完全对应 <rec> 中描述的操作
    - 若 <rec> 说“不推荐”，但 <function> 有内容 → 输出无效
    - 若 <rec> 有推荐内容，但可用功能列表中找不到对应的功能，那么输出的 <function> 应为空列表

    函数要求：
    - 在我们给的函数列表中，每个函数都有名字、描述、相似函数和参数。  
    - 每个参数有对应的描述、类型和是否必填。如果非必填，则可以省略掉；否则必须填写。
    - 如果一个参数的类型为"dict"，那么"value"中是该参数的具体结构和字段。请你确保每一个字段都要填写到；如果该参数的类型不为"dict"，且"value"不为"无法枚举"，则根据参数类型从"value"中选择一个(参数类型为str, bool时)或多个值(参数类型为list时)进行填写。
    - 不要再新增加任何函数或参数。
    - 请仔细分析指令的完整含义，考虑是否需要多个步骤或函数组合来实现该功能。如果指令需要多个函数组合完成，应按执行顺序列出所有相关函数
    - 有可能该指令并没有办法对应到合适的函数，这时候"function"字段对应空列表[]
    - 对于函数中未使用或未知的参数，保持为空


    </Output_Requirements>

    <Input_Data>
    ### 参考数据：
    {json.dumps(each_data['reference_information'], ensure_ascii=False)}

    ### 可用功能列表：
    {json.dumps(function_pool, ensure_ascii=False)}
    </Input_Data> 

    请仔细分析输入数据（特别是图片数据中的所有可见信息），基于分析框架进行推理，并严格按照JSON格式输出推荐结果。
"""
    elif 'thinking' in args.output_format:
        user_prompt = f"""
    <Role>
    你是一个专业的手机智能助手评估专家，专门分析手机主动智能推荐系统的决策过程。你需要基于多维度信息预测用户的操作意图，并判断是否应该主动向用户推荐相关功能。

    你的输出分为两个阶段：
    1. <think>：先用自然语言推理并表达你的思考过程。
    2. <function>：再将行动转化为结构化函数调用序列。

    </Role>

    <Task>
    基于以下四类输入信息进行综合分析：
    - **用户画像信息profile**：用户的个人偏好、使用习惯、历史行为模式
    - **手机环境信息phone**：当前设备状态、网络环境、电量状态等
    - **世界信息world**：时间、地点、天气、日程安排等外部环境
    - **用户行为轨迹trace**：近期的操作序列和行为模式

    你的任务是：
    1. 深度分析用户的真实需求和意图
    2. 评估需求的紧迫性和明确性
    3. 判断是否需要主动推荐
    4. 如需推荐，从给定功能池中选择合适的执行序列

    **重要说明**：
    - 只有当推荐可以通过提供的功能完全实现时，才返回功能序列
    - 如果不需要推荐或无法通过现有功能实现，返回空列表
    - 功能序列应按执行顺序排列
    </Task>

    <Analysis_Framework>
    请按以下框架进行分析：

    1. **需求识别**：从轨迹中识别用户的潜在需求，结合画像、环境、世界信息（特别是时间、地点、日程）进行交叉验证
    2. **紧迫性评估**：判断需求是否需要立即处理 —— 重点考虑当前时间、日程安排、设备状态（如低电量、弱网）是否构成紧迫性触发条件
    3. **功能匹配**：检查需求是否可通过现有功能池中的功能完全满足 —— 功能参数必须可填充且无缺失关键参数
    4. **推荐决策**：基于以上分析决定是否推荐及推荐内容 —— 仅当需求明确、紧迫、且功能可完全执行时才推荐
    </Analysis_Framework>

    <Output_Format>

        输出格式必须严格遵循：
        <think>你的思考过程</think><function>对应的结构化函数调用JSON</function>

    </Output_Format>

    <Output_Requirements>
        **<think> 部分要求：**
        思考过程应包括对用户需求的深度分析，结合给定的多维度信息进行推理。


        **<function> 部分要求（JSON格式）：**
        ```json
    {{
        "model_recommendation": [
            {{
                "name":"function_name_1",
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": ""
                }}
            }},
            {{
                "name":"function_name_2",
                "parameters": {{
                    "param1": "value1",
                    "param2": "",
                    "param3": ""
                }}
            }}
        ]
    }}
    ```

    **JSON格式要求**：
    - 所有键和字符串值必须用双引号（""）包裹
    - 键值对之间用逗号分隔，最后一个键值对后不加逗号
    - JSON字符串内部的引号使用单引号（''）
    - 禁止使用注释（// 或 /* */）
    - 每个功能必须包含 "name" 和 "parameters" 字段
    - parameters 中仅包含该功能执行所必需的参数；可选参数若无值，可设为空字符串 "" 或省略该键
    - 如无需推荐，model_recommendation字段为空数组[]


    函数要求：
    - 在我们给的函数列表中，每个函数都有名字、描述、相似函数和参数。  
    - 每个参数有对应的描述、类型和是否必填。如果非必填，则可以省略掉；否则必须填写。
    - 如果一个参数的类型为"dict"，那么"value"中是该参数的具体结构和字段。请你确保每一个字段都要填写到；如果该参数的类型不为"dict"，且"value"不为"无法枚举"，则根据参数类型从"value"中选择一个(参数类型为str, bool时)或多个值(参数类型为list时)进行填写。
    - 不要再新增加任何函数或参数。
    - 请仔细分析指令的完整含义，考虑是否需要多个步骤或函数组合来实现该功能。如果指令需要多个函数组合完成，应按执行顺序列出所有相关函数
    - 有可能该指令并没有办法对应到合适的函数，这时候"function"字段对应空列表[]
    - 对于函数中未使用或未知的参数，保持为空


    </Output_Requirements>  


    <Input_Data>
    ### 参考数据：
    {json.dumps(each_data['reference_information'], ensure_ascii=False)}

    ### 可用功能列表：
    {json.dumps(function_pool, ensure_ascii=False)}
    </Input_Data> 

    请仔细分析输入数据（特别是图片数据中的所有可见信息），基于分析框架进行推理，并严格按照JSON格式输出推荐结果。
    """    
    else:
        user_prompt = f"""
    <Role>
    你是一个专业的手机智能助手评估专家，专门分析手机主动智能推荐系统的决策过程。你需要基于多维度信息预测用户的操作意图，并判断是否应该主动向用户推荐相关功能。

    你的输出应该为结构化函数调用序列。
    </Role>

    <Task>
    基于以下四类输入信息进行综合分析：
    - **用户画像信息profile**：用户的个人偏好、使用习惯、历史行为模式
    - **手机环境信息phone**：当前设备状态、网络环境、电量状态等
    - **世界信息world**：时间、地点、天气、日程安排等外部环境
    - **用户行为轨迹trace**：近期的操作序列和行为模式

    你的任务是：
    1. 深度分析用户的真实需求和意图
    2. 评估需求的紧迫性和明确性
    3. 判断是否需要主动推荐
    4. 如需推荐，从给定功能池中选择合适的执行序列

    **重要说明**：
    - 只有当推荐可以通过提供的功能完全实现时，才返回功能序列
    - 如果不需要推荐或无法通过现有功能实现，返回空列表
    - 功能序列应按执行顺序排列
    </Task>

    <Analysis_Framework>
    请按以下框架进行分析：

    1. **需求识别**：从轨迹中识别用户的潜在需求，结合画像、环境、世界信息（特别是时间、地点、日程）进行交叉验证
    2. **紧迫性评估**：判断需求是否需要立即处理 —— 重点考虑当前时间、日程安排、设备状态（如低电量、弱网）是否构成紧迫性触发条件
    3. **功能匹配**：检查需求是否可通过现有功能池中的功能完全满足 —— 功能参数必须可填充且无缺失关键参数
    4. **推荐决策**：基于以上分析决定是否推荐及推荐内容 —— 仅当需求明确、紧迫、且功能可完全执行时才推荐
    </Analysis_Framework>


    <Output_Requirements>

        **要求输出为JSON格式：**
        ```json
    {{
        "model_recommendation": [
            {{
                "name":"function_name_1",
                "parameters": {{
                    "param1": "value1",
                    "param2": "value2",
                    "param3": ""
                }}
            }},
            {{
                "name":"function_name_2",
                "parameters": {{
                    "param1": "value1",
                    "param2": "",
                    "param3": ""
                }}
            }}
        ]
    }}
    ```

    **JSON格式要求**：
    - 所有键和字符串值必须用双引号（""）包裹
    - 键值对之间用逗号分隔，最后一个键值对后不加逗号
    - JSON字符串内部的引号使用单引号（''）
    - 禁止使用注释（// 或 /* */）
    - 每个功能必须包含 "name" 和 "parameters" 字段


    函数要求：
    - 在我们给的函数列表中，每个函数都有名字、描述、相似函数和参数。  
    - 每个参数有对应的描述、类型和是否必填。如果非必填，则可以省略掉；否则必须填写。
    - 如果一个参数的类型为"dict"，那么"value"中是该参数的具体结构和字段。请你确保每一个字段都要填写到；如果该参数的类型不为"dict"，且"value"不为"无法枚举"，则根据参数类型从"value"中选择一个(参数类型为str, bool时)或多个值(参数类型为list时)进行填写。
    - 不要再新增加任何函数或参数。
    - 请仔细分析指令的完整含义，考虑是否需要多个步骤或函数组合来实现该功能。如果指令需要多个函数组合完成，应按执行顺序列出所有相关函数
    - 有可能该指令并没有办法对应到合适的函数，这时候"function"字段对应空列表[]
    - 对于函数中未使用或未知的参数，保持为空


    </Output_Requirements>

    <Input_Data>
    ### 参考数据：
    {json.dumps(each_data['reference_information'], ensure_ascii=False)}

    ### 可用功能列表：
    {json.dumps(function_pool, ensure_ascii=False)}
    </Input_Data> 

    请仔细分析输入数据（特别是图片数据中的所有可见信息），基于分析框架进行推理，并严格按照JSON格式输出推荐结果。
"""
    content_list = [{"type": "text", "text": user_prompt}]

    # 将content_list合并为单个纯文本prompt
    prompt_text = "\n".join([c.get("text", "") for c in content_list if c.get("type") == "text"])
    return prompt_text


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_to_json(actions, filename):
    """Append *actions* to a JSON list stored in *filename*."""
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(actions)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {filename}")


def extract_json_from_text(text):
    """Try to extract a JSON object from *text*; return -1 on failure."""
    json_block_match = re.search(r'```(?:json|JSON|)\s*\n?({.*?})\s*```', text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError):
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx != -1 and end_idx != 0:
        json_candidate = text[start_idx:end_idx]
        try:
            return ast.literal_eval(json_candidate)
        except (ValueError, SyntaxError):
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    return -1

def check_each_data(recommendation, each_data):
    """Best-match metrics against all reference intents — identical to the evaluation
    protocol used in the paper. Returns (acc, precision, recall, f1, false_alarm,
    true_negative) for the best-matching intent, plus that intent's index.

    false_alarm / true_negative are the FTR bookkeeping (None when not applicable):
      - gold intent non-empty -> false_alarm=None, true_negative=(pred empty)
      - gold intent empty      -> false_alarm=(pred non-empty), true_negative=None
    """
    metric_group = []
    for intent in each_data['recommendations']:
        std_func_list = [item['name'] for item in intent['function']]
        acc = 0
        if recommendation == ['None'] or isinstance(recommendation, str):
            metric_group.append((0, 0, 0, 0, None, None))
            continue

        if len(recommendation) == len(std_func_list) and all(r == s for r, s in zip(recommendation, std_func_list)):
            acc = 1
        else:
            acc = 0
        rec_list = list(recommendation) if not isinstance(recommendation, (set, frozenset)) else list(recommendation)
        std_set = set(std_func_list)
        rec_set = set(rec_list)
        intersection = rec_set & std_set

        if len(std_func_list) == 0:
            recall = int(len(rec_list) == 0)
        else:
            recall = len(intersection) / len(std_set)

        if len(rec_set) == 0:
            precision = int(len(std_func_list) == 0)
        else:
            precision = len(intersection) / len(rec_set)

        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        if len(std_set) != 0:
            false_alarm = None
            true_negative = (len(rec_list) == 0)
        else:
            false_alarm = (len(rec_list) > 0)
            true_negative = None

        recommendation = rec_list  # restore list for next iteration

        metric_group.append((acc, precision, recall, f1, false_alarm, true_negative))

    if metric_group:
        best_result = max(metric_group, key=lambda x: (x[0], x[3]))
        best_id = metric_group.index(best_result)
    else:
        best_result = (0, 0.0, 0, 0, None, None)
        best_id = 0
    return best_result, best_id


# ---------------------------------------------------------------------------
# Helper: classify a data sample as GUI or TEXT based on trace type
# ---------------------------------------------------------------------------

def _is_gui_data(each_data: Dict) -> bool:
    trace = each_data['reference_information']['trace']
    return len(trace) > 0 and isinstance(trace[0], dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Image collection + output parsing (shared by inference)
# ---------------------------------------------------------------------------

def image_to_base64_str(image_path: str, max_pixels: int = None) -> str:
    """Convert an image file to a base64-encoded string, optionally down-sampling."""
    if not PIL_AVAILABLE or max_pixels is None:
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            total_pixels = w * h

            if total_pixels > max_pixels:
                scale_ratio = (max_pixels / total_pixels) ** 0.5
                new_w = int(w * scale_ratio)
                new_h = int(h * scale_ratio)

                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

                buffered = io.BytesIO()
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                     img_resized = img_resized.convert('RGB')

                img_resized.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
            else:
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        print(f"Image processing failed, falling back to raw read: {e}")
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def _collect_images(each_data, args):
    imgs = []
    try:
        trace = each_data['reference_information']['trace']
        if isinstance(trace, list) and len(trace) > 0 and isinstance(trace[0], dict) \
                and trace[0].get('source') == 'picture':
            for step in trace:
                p = step.get('picture')
                if p:
                    if args.image_prefix and p.startswith(args.image_prefix):
                        full = os.path.join(args.image_root, p[len(args.image_prefix):])
                    elif args.image_root:
                        full = os.path.join(args.image_root, p)
                    else:
                        full = p
                    if os.path.exists(full):
                        imgs.append(image_to_base64_str(full, args.image_max_pixels))
    except Exception as e:
        print(f"Image collection failed: {e}")
    return imgs


def _parse_output(text, args):
    """Parse one raw model output into a function list (same logic as the paper's evaluation)."""
    try:
        if args.output_format:
            m = re.search(r'<function>(.*?)</function>', text, re.DOTALL)
            function = m.group(1).strip() if m else text
        else:
            function = text
        result = extract_json_from_text(function)
        if result == -1:
            return text, str(function)
        if isinstance(result, dict) and 'model_recommendation' in result:
            try:
                json.dumps(result)
                return text, result['model_recommendation']
            except (TypeError, ValueError):
                return text, str(function)
        return text, str(function)
    except Exception as e:
        print(f"Output parsing failed: {e}")
        return text, ["None"]


def _sample_id(each, idx):
    return each.get('benchmark_metadata', {}).get('id') or f"__idx_{idx}"


def run_eval_vllm(args):
    """Local in-process vLLM inference. Mirrors the paper's evaluation
    pipeline: pass {"prompt": text, "image": [base64...]} to llm.generate(), read
    out.outputs[0].text, parse with _parse_output."""
    from vllm import LLM, SamplingParams

    model_name = args.model_path.split('/')[-1]
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, model_name + '.json')
    ckpt_file = output_file + '.ckpt.json'

    with open(args.eval_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    with open(args.function_pool_file, 'r', encoding='utf-8') as f:
        function_pool = json.load(f)

    cache = {}
    if os.path.exists(ckpt_file):
        try:
            cache = json.load(open(ckpt_file, encoding='utf-8'))
            print(f"Resuming from checkpoint: {len(cache)} samples already done", flush=True)
        except Exception:
            cache = {}

    todo = [(i, d) for i, d in enumerate(datas) if _sample_id(d, i) not in cache]
    print(f"Total {len(datas)}  done {len(cache)}  todo {len(todo)}", flush=True)

    llm_kwargs = dict(model=args.model_path, trust_remote_code=True,
                      tensor_parallel_size=args.tensor_parallel_size,
                      pipeline_parallel_size=args.pipeline_parallel_size,
                      gpu_memory_utilization=args.gpu_memory_utilization)
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                     max_tokens=args.max_tokens)

    # Build batch prompts exactly as in the paper's evaluation.
    batch_prompts, order = [], []
    for idx, each in todo:
        prompt_text = build_prompt_text(each, function_pool, args)
        imgs = _collect_images(each, args)
        if len(imgs) == 0:
            batch_prompts.append({"prompt": prompt_text})
        elif len(imgs) == 1:
            batch_prompts.append({"prompt": prompt_text, "image": imgs[0]})
        else:
            batch_prompts.append({"prompt": prompt_text, "image": imgs})
        order.append((idx, each))

    t0 = time.time()
    outputs = llm.generate(batch_prompts, sampling_params)
    print(f"vLLM generate done in {time.time()-t0:.0f}s for {len(outputs)} samples", flush=True)

    for (idx, each), out in zip(order, outputs):
        text = out.outputs[0].text
        _, func = _parse_output(text, args)
        cache[_sample_id(each, idx)] = {"text": text, "func": func}
    json.dump(cache, open(ckpt_file, 'w', encoding='utf-8'), ensure_ascii=False)

    # assemble final output in original order
    processed, badcase = [], []
    for i, each in enumerate(datas):
        entry = cache.get(_sample_id(each, i), {"text": "", "func": ["None"]})
        text, func = entry["text"], entry["func"]
        if func is None:
            func = ["None"]
        bm = each.setdefault('benchmark_metadata', {})
        bm.setdefault('model_output', {})[model_name] = text
        bm.setdefault('model_function_result', {})[model_name] = func
        processed.append(each)
        if func == ['None'] or isinstance(func, str):
            badcase.append(each)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_path, model_name + '_badcase.json'), 'w', encoding='utf-8') as f:
        json.dump(badcase, f, ensure_ascii=False, indent=2)
    print(f"Inference done -> {output_file}  (samples {len(processed)})")


def run_compute(args):
    model_name = args.model_path.split('/')[-1]
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, model_name + '.json')
    badcase_file = os.path.join(args.output_path, model_name + '_badcase.json')
    wrongcase_file = os.path.join(args.output_path, model_name + '_wrongcase.json')
    correctcase_file = os.path.join(args.output_path, model_name + '_correctcase.json')
    log_file = os.path.join(args.output_path, 'eval_logs', model_name + '.txt')
    with open(args.function_pool_file, 'r', encoding='utf-8') as f:
        function_pool = json.load(f)
    wrongcase_data = []
    correctcase_data = []

    with open(output_file, 'r', encoding='utf-8') as file:
        datas = json.load(file)

    extraction_failed = 0
    total = len(datas)
    total_acc, total_precision, total_recall, total_f1 = 0, 0, 0, 0
    difficulty_acc = {1: 0, 2: 0, 3: 0}
    difficulty_precision = {1: 0, 2: 0, 3: 0}
    difficulty_recall = {1: 0, 2: 0, 3: 0}
    difficulty_f1 = {1: 0, 2: 0, 3: 0}
    difficulty_num = {1: 0, 2: 0, 3: 0}
    difficulty_false_alarm = []      # list of (difficulty, bool); FTR bookkeeping
    difficulty_true_negative = []    # list of (difficulty, bool); miss-rate bookkeeping

    text_total_acc, text_total_precision, text_total_recall, text_total_f1 = 0, 0, 0, 0
    text_difficulty_acc = {1: 0, 2: 0, 3: 0}
    text_difficulty_precision = {1: 0, 2: 0, 3: 0}
    text_difficulty_recall = {1: 0, 2: 0, 3: 0}
    text_difficulty_f1 = {1: 0, 2: 0, 3: 0}
    text_difficulty_num = {1: 0, 2: 0, 3: 0}
    text_difficulty_false_alarm = []
    text_difficulty_true_negative = []
    text_extraction_failed = 0

    gui_total_acc, gui_total_precision, gui_total_recall, gui_total_f1 = 0, 0, 0, 0
    gui_difficulty_acc = {1: 0, 2: 0, 3: 0}
    gui_difficulty_precision = {1: 0, 2: 0, 3: 0}
    gui_difficulty_recall = {1: 0, 2: 0, 3: 0}
    gui_difficulty_f1 = {1: 0, 2: 0, 3: 0}
    gui_difficulty_num = {1: 0, 2: 0, 3: 0}
    gui_difficulty_false_alarm = []
    gui_difficulty_true_negative = []
    gui_extraction_failed = 0

    for each_data in datas:
        if each_data['benchmark_metadata']['model_function_result'][model_name] == ['None'] or \
                isinstance(each_data['benchmark_metadata']['model_function_result'][model_name], str):
            recommendation = ['None']
            extraction_failed += 1
            if _is_gui_data(each_data):
                gui_extraction_failed += 1
            else:
                text_extraction_failed += 1
        else:
            try:
                recommendation = [item['name'] for item in each_data['benchmark_metadata']["model_function_result"][model_name]]
            except Exception:
                recommendation = ['None']
                extraction_failed += 1
                if _is_gui_data(each_data):
                    gui_extraction_failed += 1
                else:
                    text_extraction_failed += 1

        difficulty = each_data['benchmark_metadata']['difficulty_level']
        if difficulty not in [1, 2, 3]:
            difficulty = 1
        difficulty_num[difficulty] += 1

        (acc, precision, recall, f1, false_alarm, true_negative), index = check_each_data(recommendation, each_data)
        each_data['benchmark_metadata']['best_matched_id'] = index

        if not (recommendation == ['None'] or isinstance(recommendation, str)):
            if acc == 1:
                correctcase_data.append(each_data)
            else:
                wrongcase_data.append(each_data)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        difficulty_precision[difficulty] += precision
        difficulty_recall[difficulty] += recall
        difficulty_f1[difficulty] += f1
        if false_alarm is not None:
            difficulty_false_alarm.append((difficulty, false_alarm))
        if true_negative is not None:
            difficulty_true_negative.append((difficulty, true_negative))

        if _is_gui_data(each_data):
            gui_total_precision += precision
            gui_total_recall += recall
            gui_total_f1 += f1
            gui_difficulty_precision[difficulty] += precision
            gui_difficulty_recall[difficulty] += recall
            gui_difficulty_f1[difficulty] += f1
            gui_difficulty_num[difficulty] += 1
            if false_alarm is not None:
                gui_difficulty_false_alarm.append((difficulty, false_alarm))
            if true_negative is not None:
                gui_difficulty_true_negative.append((difficulty, true_negative))
        else:
            text_total_precision += precision
            text_total_recall += recall
            text_total_f1 += f1
            text_difficulty_precision[difficulty] += precision
            text_difficulty_recall[difficulty] += recall
            text_difficulty_f1[difficulty] += f1
            text_difficulty_num[difficulty] += 1
            if false_alarm is not None:
                text_difficulty_false_alarm.append((difficulty, false_alarm))
            if true_negative is not None:
                text_difficulty_true_negative.append((difficulty, true_negative))

    total_acc = len(correctcase_data)

    for each_data in correctcase_data:
        difficulty = each_data['benchmark_metadata']['difficulty_level']
        if difficulty not in [1, 2, 3]:
            difficulty = 1
        difficulty_acc[difficulty] += 1
        if _is_gui_data(each_data):
            gui_total_acc += 1
            gui_difficulty_acc[difficulty] += 1
        else:
            text_total_acc += 1
            text_difficulty_acc[difficulty] += 1

    with open(wrongcase_file, 'w', encoding='utf-8') as f:
        json.dump(wrongcase_data, f, ensure_ascii=False, indent=2)
    with open(correctcase_file, 'w', encoding='utf-8') as f:
        json.dump(correctcase_data, f, ensure_ascii=False, indent=2)

    if total == 0:
        print("No valid evaluation data found. Please run 'eval' mode first.")
        sys.exit(1)

    overall_accuracy = total_acc / total
    overall_precision = total_precision / total
    overall_recall = total_recall / total
    overall_f1 = total_f1 / total

    # match the paper's evaluation exactly: empty-trace samples count as neither
    text_data = [d for d in datas
                 if len(d['reference_information']['trace']) > 0
                 and isinstance(d['reference_information']['trace'][0], str)]
    gui_data = [d for d in datas
                if len(d['reference_information']['trace']) > 0
                and isinstance(d['reference_information']['trace'][0], dict)]

    def _rate(pairs, diff=None):
        vals = [p[1] for p in pairs if (diff is None or p[0] == diff)]
        return sum(vals) / len(vals) if vals else None

    overall_ftr = _rate(difficulty_false_alarm)
    overall_mr = _rate(difficulty_true_negative)

    print(f"\n{args.model_path} — Overall Results:")
    print(f"  Accuracy  : {overall_accuracy:.5f}")
    print(f"  Precision : {overall_precision:.5f}")
    print(f"  Recall    : {overall_recall:.5f}")
    print(f"  F1        : {overall_f1:.5f}")
    if overall_ftr is not None:
        print(f"  FTR       : {overall_ftr:.5f}  (误报率)")
    if overall_mr is not None:
        print(f"  MissRate  : {overall_mr:.5f}  (漏报率)")
    for diff in [1, 2, 3]:
        if difficulty_num[diff] > 0:
            print(f"\n  Difficulty {diff} ({difficulty_num[diff]} samples):")
            print(f"    Accuracy  : {difficulty_acc[diff] / difficulty_num[diff]:.5f}")
            print(f"    Precision : {difficulty_precision[diff] / difficulty_num[diff]:.5f}")
            print(f"    Recall    : {difficulty_recall[diff] / difficulty_num[diff]:.5f}")
            print(f"    F1        : {difficulty_f1[diff] / difficulty_num[diff]:.5f}")
            _f = _rate(difficulty_false_alarm, diff)
            if _f is not None:
                print(f"    FTR       : {_f:.5f}")

    print(f"\n{'=' * 40}")
    print(f"TEXT Results ({len(text_data)} samples)")
    print(f"{'=' * 40}")
    if len(text_data) > 0:
        text_overall_accuracy = text_total_acc / len(text_data)
        text_overall_precision = text_total_precision / len(text_data)
        text_overall_recall = text_total_recall / len(text_data)
        text_overall_f1 = text_total_f1 / len(text_data)
        print(f"  Accuracy  : {text_overall_accuracy:.5f}")
        print(f"  Precision : {text_overall_precision:.5f}")
        print(f"  Recall    : {text_overall_recall:.5f}")
        print(f"  F1        : {text_overall_f1:.5f}")
        _tf = _rate(text_difficulty_false_alarm)
        if _tf is not None:
            print(f"  FTR       : {_tf:.5f}  (误报率)")
        for diff in [1, 2, 3]:
            if text_difficulty_num[diff] > 0:
                print(f"\n  Difficulty {diff} ({text_difficulty_num[diff]} samples):")
                print(f"    Accuracy  : {text_difficulty_acc[diff] / text_difficulty_num[diff]:.5f}")
                print(f"    Precision : {text_difficulty_precision[diff] / text_difficulty_num[diff]:.5f}")
                print(f"    Recall    : {text_difficulty_recall[diff] / text_difficulty_num[diff]:.5f}")
                print(f"    F1        : {text_difficulty_f1[diff] / text_difficulty_num[diff]:.5f}")
                _f = _rate(text_difficulty_false_alarm, diff)
                if _f is not None:
                    print(f"    FTR       : {_f:.5f}")
        print(f"  Extraction failures: {text_extraction_failed}")

    print(f"\n{'=' * 40}")
    print(f"GUI Results ({len(gui_data)} samples)")
    print(f"{'=' * 40}")
    if len(gui_data) > 0:
        gui_overall_accuracy = gui_total_acc / len(gui_data)
        gui_overall_precision = gui_total_precision / len(gui_data)
        gui_overall_recall = gui_total_recall / len(gui_data)
        gui_overall_f1 = gui_total_f1 / len(gui_data)
        print(f"  Accuracy  : {gui_overall_accuracy:.5f}")
        print(f"  Precision : {gui_overall_precision:.5f}")
        print(f"  Recall    : {gui_overall_recall:.5f}")
        print(f"  F1        : {gui_overall_f1:.5f}")
        _gf = _rate(gui_difficulty_false_alarm)
        if _gf is not None:
            print(f"  FTR       : {_gf:.5f}  (误报率)")
        for diff in [1, 2, 3]:
            if gui_difficulty_num[diff] > 0:
                print(f"\n  Difficulty {diff} ({gui_difficulty_num[diff]} samples):")
                print(f"    Accuracy  : {gui_difficulty_acc[diff] / gui_difficulty_num[diff]:.5f}")
                print(f"    Precision : {gui_difficulty_precision[diff] / gui_difficulty_num[diff]:.5f}")
                print(f"    Recall    : {gui_difficulty_recall[diff] / gui_difficulty_num[diff]:.5f}")
                print(f"    F1        : {gui_difficulty_f1[diff] / gui_difficulty_num[diff]:.5f}")
                _f = _rate(gui_difficulty_false_alarm, diff)
                if _f is not None:
                    print(f"    FTR       : {_f:.5f}")
        print(f"  Extraction failures: {gui_extraction_failed}")

    # Write log file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Report - {args.model_path}\n")
        f.write(f"\nOverall Results:\n")
        f.write(f"  Accuracy  : {overall_accuracy:.5f}\n")
        f.write(f"  Precision : {overall_precision:.5f}\n")
        f.write(f"  Recall    : {overall_recall:.5f}\n")
        f.write(f"  F1        : {overall_f1:.5f}\n")
        if overall_ftr is not None:
            f.write(f"  FTR       : {overall_ftr:.5f}\n")
        if overall_mr is not None:
            f.write(f"  MissRate  : {overall_mr:.5f}\n")

        for diff in [1, 2, 3]:
            if difficulty_num[diff] > 0:
                f.write(f"\n  Difficulty {diff} ({difficulty_num[diff]} samples):\n")
                f.write(f"    Accuracy  : {difficulty_acc[diff] / difficulty_num[diff]:.5f}\n")
                f.write(f"    Precision : {difficulty_precision[diff] / difficulty_num[diff]:.5f}\n")
                f.write(f"    Recall    : {difficulty_recall[diff] / difficulty_num[diff]:.5f}\n")
                f.write(f"    F1        : {difficulty_f1[diff] / difficulty_num[diff]:.5f}\n")
                _f = _rate(difficulty_false_alarm, diff)
                if _f is not None:
                    f.write(f"    FTR       : {_f:.5f}\n")
        f.write(f"\n  Extraction failures: {extraction_failed}\n")

        if len(text_data) > 0:
            f.write(f"\n{'=' * 40}\n")
            f.write(f"TEXT Results ({len(text_data)} samples)\n")
            f.write(f"  Accuracy  : {text_overall_accuracy:.5f}\n")
            f.write(f"  Precision : {text_overall_precision:.5f}\n")
            f.write(f"  Recall    : {text_overall_recall:.5f}\n")
            f.write(f"  F1        : {text_overall_f1:.5f}\n")
            _tf = _rate(text_difficulty_false_alarm)
            if _tf is not None:
                f.write(f"  FTR       : {_tf:.5f}\n")
            for diff in [1, 2, 3]:
                if text_difficulty_num[diff] > 0:
                    f.write(f"\n  Difficulty {diff} ({text_difficulty_num[diff]} samples):\n")
                    f.write(f"    Accuracy  : {text_difficulty_acc[diff] / text_difficulty_num[diff]:.5f}\n")
                    f.write(f"    Precision : {text_difficulty_precision[diff] / text_difficulty_num[diff]:.5f}\n")
                    f.write(f"    Recall    : {text_difficulty_recall[diff] / text_difficulty_num[diff]:.5f}\n")
                    f.write(f"    F1        : {text_difficulty_f1[diff] / text_difficulty_num[diff]:.5f}\n")
                    _f = _rate(text_difficulty_false_alarm, diff)
                    if _f is not None:
                        f.write(f"    FTR       : {_f:.5f}\n")
            f.write(f"\n  Extraction failures: {text_extraction_failed}\n")

        if len(gui_data) > 0:
            f.write(f"\n{'=' * 40}\n")
            f.write(f"GUI Results ({len(gui_data)} samples)\n")
            f.write(f"  Accuracy  : {gui_overall_accuracy:.5f}\n")
            f.write(f"  Precision : {gui_overall_precision:.5f}\n")
            f.write(f"  Recall    : {gui_overall_recall:.5f}\n")
            f.write(f"  F1        : {gui_overall_f1:.5f}\n")
            _gf = _rate(gui_difficulty_false_alarm)
            if _gf is not None:
                f.write(f"  FTR       : {_gf:.5f}\n")
            for diff in [1, 2, 3]:
                if gui_difficulty_num[diff] > 0:
                    f.write(f"\n  Difficulty {diff} ({gui_difficulty_num[diff]} samples):\n")
                    f.write(f"    Accuracy  : {gui_difficulty_acc[diff] / gui_difficulty_num[diff]:.5f}\n")
                    f.write(f"    Precision : {gui_difficulty_precision[diff] / gui_difficulty_num[diff]:.5f}\n")
                    f.write(f"    Recall    : {gui_difficulty_recall[diff] / gui_difficulty_num[diff]:.5f}\n")
                    f.write(f"    F1        : {gui_difficulty_f1[diff] / gui_difficulty_num[diff]:.5f}\n")
                    _f = _rate(gui_difficulty_false_alarm, diff)
                    if _f is not None:
                        f.write(f"    FTR       : {_f:.5f}\n")
            f.write(f"\n  Extraction failures: {gui_extraction_failed}\n")

    print(f"\nEvaluation log saved to: {log_file}")


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "eval":
        run_eval_vllm(args)
    else:
        run_compute(args)

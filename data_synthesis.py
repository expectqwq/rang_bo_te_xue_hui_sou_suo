import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from fake_search import FakeSearch, SEARCH_TOOL

class DataSynthesizer:
    def __init__(self, model_path="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.search_tool = FakeSearch()
        
    def generate_response(self, query):
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个智能助手，具备搜索能力。\n\n"
                    "你的任务是在用户提出问题时，先判断是否可以直接回答。如果你非常确定答案来自你的已知知识库，并且信息是稳定的历史事实，请直接用自然语言简洁回答。\n\n"
                    "但如果你对答案不确定，或者问题涉及：\n"
                    "- 最新事件、动态、新闻、人物、职位变更、科技发展\n"
                    "- 需要获取互联网信息（如实时数据、当前职位、排行榜等）\n"
                    "- 涉及未来预测、评估、最新趋势等\n\n"
                    "此时你必须调用一个名为 `search` 的工具。\n\n"
                    "调用格式如下（必须严格保持 JSON 格式，**不加解释、不加 markdown 包裹、不输出多余内容**）：\n"
                    "{\n"
                    "  \"name\": \"search\",\n"
                    "  \"arguments\": {\n"
                    "    \"keyword\": \"用一句话精准描述用户的问题\",\n"
                    "    \"top_k\": 3\n"
                    "  }\n"
                    "}\n\n"
                    "必须遵守：\n"
                    "- 如果调用工具，**只输出 JSON 格式调用**，不添加任何解释或注释\n"
                    "- 如果不调用工具，**直接自然语言回答**，不要包含任何 JSON 或代码块\n"
                    "- 严格保持格式准确，避免输出错误结构或多余内容\n"
                )
            },
            {"role": "user", "content": query}
        ]

        tools = [SEARCH_TOOL]
        
        input_text = f"<|im_start|>system\n{messages[0]['content']}\n\n可用工具：{json.dumps(tools, ensure_ascii=False)}<|im_end|>\n<|im_start|>user\n{messages[1]['content']}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入，获取 input_ids 和 attention_mask
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        # 生成回答，显式传入 attention_mask
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=512,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回答
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        response = response.strip()
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        return response
    
    def process_with_search(self, query, response):
        """处理包含搜索的对话"""
        # 解析是否包含工具调用
        if "search" in response and "keyword" in response:
            # 提取搜索关键词
            try:
                # 简单的关键词提取逻辑
                start_idx = response.find('"keyword":')
                if start_idx != -1:
                    start_idx = response.find('"', start_idx + 10) + 1
                    end_idx = response.find('"', start_idx)
                    keyword = response[start_idx:end_idx]
                    
                    # 执行搜索
                    search_results = self.search_tool.search(keyword, 3)
                    
                    # 构建包含搜索结果的新对话
                    search_result_text = "\n".join([f"结果{i+1}: {result}" for i, result in enumerate(search_results)])
                    
                    final_messages = [
                        {"role": "system", "content": "你是一个智能助手，能够进行复杂思考并使用搜索工具获取信息。"},
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": response},
                        {"role": "tool", "content": search_result_text, "tool_call_id": "search_1"},
                        {"role": "assistant", "content": ""}
                    ]
                    
                    # 生成最终回答
                    input_text = self.format_messages_for_generation(final_messages[:-1])
                    # 编码输入，获取 input_ids 和 attention_mask
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.model.device)

                    # 生成回答，显式传入 attention_mask
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=512,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # 解码回答
                    final_response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                    if "<|im_end|>" in final_response:
                        final_response = final_response.split("<|im_end|>")[0]
                    final_messages[-1]["content"] = final_response
                    
                    return final_messages
            except Exception as e:
                print(f"搜索处理错误: {e}")
        
        return None
    
    def format_messages_for_generation(self, messages):
        formatted = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                formatted += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                formatted += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "tool":
                formatted += f"<|im_start|>tool\n{msg['content']}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
        return formatted
    
def load_questions_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def synthesize_data():
    synthesizer = DataSynthesizer()

    print("正在合成数据，请稍候...")
    
    questions_with_search = load_questions_from_file("question_with_search.txt")
    questions_without_search = load_questions_from_file("question_without_search.txt")
    
    synthetic_data = []
    
    # 合并所有问题，让模型自行判断是否需要搜索
    all_questions = questions_with_search + questions_without_search
    # all_questions = all_questions[:5]
    
    for question in questions_without_search:
        print("处理问题:", question, flush=True)
        for _ in range(1):
            try:
                # 生成初始回答
                initial_response = synthesizer.generate_response(question)
                
                # 处理搜索逻辑
                search_conversation = synthesizer.process_with_search(question, initial_response)
                
                if search_conversation:
                    # 如果成功处理了搜索，添加完整的对话数据
                    synthetic_data.append({
                        "conversation": search_conversation,
                        "type": "with_search"
                    })
                    break
                    
            except Exception as e:
                print(f"处理问题时出错: {question[:50]}... - {e}")
                continue
        else:
            # 如果没有触发搜索，添加基本对话
            synthetic_data.append({
                "conversation": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": initial_response}
                ],
                "type": "direct_answer"
            })

    # 保存数据
    with open("synthetic_data.json", "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
    
    print(f"合成数据完成，共生成 {len(synthetic_data)} 条数据")
    return synthetic_data

if __name__ == "__main__":
    synthesize_data()
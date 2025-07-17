import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fake_search import FakeSearch
import re
import json

class ChatBot:
    def __init__(self, model_path="./new_lora_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            "/remote-home1/share/models/Qwen2.5-0.5B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        self.search_tool = FakeSearch()

    def should_search(self, response, messages):
        # 如果消息中已经包含tool角色，说明已经在处理搜索结果，不再搜索
        for msg in messages:
            if msg["role"] == "system" and msg["content"] == (
                        "你是一个智能助手，已经调用过搜索工具并获得了结果。"
                        "接下来你不能再进行搜索，只能完全基于现有搜索结果，给出准确、有用、简洁的回答。"
                        "不要重复调用搜索，也不要回答‘我需要搜索’之类的内容。"
                        "请聚焦用户问题，从已知信息中提取答案，给出合理的回答。"
                    ):
                return False
        
        # 检查是否包含搜索指令格式
        if "</think>" in response:
            response = response.split("</think>")[1]
        response = response.strip()
        response = response.removeprefix("```json")
        response = response.removesuffix('```')

        try:
            obj = json.loads(response)
            return obj.get("name") == "search"
        except Exception:
            return False
        
    def generate_response(self, messages):
        # 构建输入
        input_text = ""
        for msg in messages:
            if msg["role"] == "system":
                input_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                input_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                input_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "tool":
                input_text += f"<|im_start|>tool\n{msg['content']}<|im_end|>\n"
        
        input_text += "<|im_start|>assistant\n"
        
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
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回答
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        
        if self.should_search(response, messages):
            return self.handle_search(response, messages)
        
        return response
    
    def get_keyword(self, response):
        if "</think>" in response:
            response = response.split("</think>")[1]
        response = response.strip()
        response = response.removeprefix("```json")
        response = response.removesuffix('```')

        start_idx = response.find('"keyword":')
        if start_idx != -1:
            start_idx = response.find('"', start_idx + 10) + 1
            end_idx = response.find('"', start_idx)
            return response[start_idx:end_idx]
        return None
    
    def handle_search(self, response, messages):
        try:
            keyword = self.get_keyword(response)
            print("正在搜索：", keyword, flush = True)
            search_results = self.search_tool.search(keyword, 3)
            tool_text = "\n".join(search_results)
            print(tool_text, flush = True)
            
            summary_system = {
                "role": "system",
                "content": (
                    "你是一个智能助手，已经调用过搜索工具并获得了结果。"
                    "接下来你不能再进行搜索，只能完全基于现有搜索结果，给出准确、有用、简洁的回答。"
                    "不要重复调用搜索，也不要回答‘我需要搜索’之类的内容。"
                    "请聚焦用户问题，从已知信息中提取答案，给出合理的回答。"
                )
            }

            return self.generate_response([summary_system, messages[1], {"role": "tool", "content": tool_text}])
        except Exception as e:
            return f"搜索处理出错: {str(e)}"
    
    def chat(self, user_input):
        messages = [{"role": "system", "content": "你是一个智能助手，能够进行复杂思考并使用搜索工具获取信息。"}]
            
        messages.append({"role": "user", "content": user_input})
            
        response = self.generate_response(messages.copy())

        print("=" * 50)
        print(f"助手:\n{response}")
        print("=" * 50)
        print("", flush = True)
            
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.chat("雀魂最新的角色是谁？")
    chatbot.chat("大象是一种鱼吗？")
    chatbot.chat("python是什么？")
    chatbot.chat("最新的电影有哪些？")
    chatbot.chat("天空是什么颜色的？")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class FakeSearch:
    def __init__(self, model_path="/remote-home1/share/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        print("正在加载搜索模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("搜索模型加载完成！")
    
    def chat(self, messages: list):
        # 构建输入文本
        input_text = ""
        for msg in messages:
            if msg["role"] == "user":
                input_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "system":
                input_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                input_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        
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
                max_new_tokens=2048,
                temperature=0.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码回答
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        # 模拟OpenAI API返回格式
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, content):
                self.message = MockResponse(content)
        
        class MockResult:
            def __init__(self, content):
                self.choices = [MockChoice(content)]
        
        return MockResult(response)
    
    def search(self, keyword, top_k=3):
        # 构建搜索提示
        search_prompt = f"""
        请你扮演一个专业搜索引擎助手，收到以下关键词后，务必严格按照下面的格式输出搜索结果：

        1. 每条结果必须使用“1.”、“2.”、……编号。
        2. 每条结果独立成段，段落之间用一个空行分隔。
        3. 每条结果不超过200字，只包含信息，不得有多余的提示、引言或结尾。
        4. 只输出编号结果，不要输出除结果外的任何文字。
        5. 必须给出具体的结果，必要时可以编造一些合理的内容，但必须符合搜索引擎的常规输出格式。

        关键词：{keyword}
        请给出{min(top_k,10)}条搜索结果：
        """
        
        res = self.chat([{
            "role": "user",
            "content": search_prompt
        }])
        
        # 处理响应内容
        content = res.choices[0].message.content
        
        # 移除thinking标签内容
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        
        # 按空行分割结果
        results = content.split('\n\n')
        return results[:top_k]

# 工具定义
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索引擎，在需要获取最新信息、实时数据、特定事实或用户询问需要网络搜索才能回答的问题时需要调用此工具",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "使用搜索引擎所需的关键词"},
                "top_k": {"type": "number", "default": 3, "description": "返回的搜索结果数量"}
            },
            "required": ["keyword"]
        }
    }
}

if __name__ == "__main__":
    search_tool = FakeSearch()
    search_results = search_tool.search("今天上海的天气")
    print("Search Results:", search_results)
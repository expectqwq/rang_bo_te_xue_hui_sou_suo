import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import json
import random

class SplitChatDataset:
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        
        # 拆分数据
        self.processed_data = self._split_search_data()
        random.shuffle(self.processed_data)  # 打乱数据顺序
    
    def _split_search_data(self):
        """将搜索任务拆分成两个独立的训练样本"""
        processed_samples = []
        
        for item in self.data:
            messages = item["messages"]
            
            if len(messages) >= 5:
                # 搜索任务的完整结构应该是：
                # [system, user, assistant(工具调用), tool(搜索结果), assistant(最终回答)]
                
                system_msg = messages[0]
                user_msg = messages[1]
                tool_call_msg = messages[2]  # 工具调用
                tool_result_msg = messages[3]  # 搜索结果
                final_answer_msg = messages[4]  # 最终回答
                
                # 样本1: 学习工具调用
                sample1_messages = [system_msg, user_msg, tool_call_msg]
                processed_samples.append({
                    "messages": sample1_messages,
                    "sample_type": "tool_call"
                })
                
                # 样本2: 学习利用工具结果进行总结
                # 构建包含搜索结果的上下文
                summary_system = {
                    "role": "system",
                    "content": (
                        "你是一个智能助手，已经调用过搜索工具并获得了结果。"
                        "接下来你不能再进行搜索，只能完全基于现有搜索结果，给出准确、有用、简洁的回答。"
                        "不要重复调用搜索，也不要回答‘我需要搜索’之类的内容。"
                        "请聚焦用户问题，从已知信息中提取答案，给出合理的回答。"
                    )
                }
                
                sample2_messages = [summary_system, user_msg, tool_result_msg, final_answer_msg]
                processed_samples.append({
                    "messages": sample2_messages,
                    "sample_type": "tool_summary"
                })                
                processed_samples.append({
                    "messages": sample2_messages,
                    "sample_type": "tool_summary"
                })
                
            else:
                # 非搜索任务保持原样
                processed_samples.append({
                    "messages": messages,
                    "sample_type": "direct_answer"
                })
                processed_samples.append({
                    "messages": messages,
                    "sample_type": "direct_answer"
                })                
                processed_samples.append({
                    "messages": messages,
                    "sample_type": "direct_answer"
                })
        
        return processed_samples
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        messages = item["messages"]
        
        # 构建对话文本
        conversation = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "tool":
                conversation += f"<|im_start|>tool\n{msg['content']}<|im_end|>\n"
        
        # 分离输入和目标
        parts = conversation.split("<|im_start|>assistant\n")

        if len(parts) < 2:
            input_text = conversation
            target_text = ""
        else:
            # 找到最后一个assistant回答作为目标
            input_text = parts[0] + "<|im_start|>assistant\n"
            target_text = parts[-1]
        
        # 编码
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, truncation=True)
        target_ids = self.tokenizer.encode(target_text, max_length=self.max_length, truncation=True)
        
        # 构建标签，只对assistant部分计算loss
        labels = [-100] * len(input_ids) + target_ids
        input_ids = input_ids + target_ids
        
        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # 填充
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_model():
    # 加载模型和tokenizer
    model_path = "/remote-home1/share/models/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # LoRA配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备数据 - 使用拆分后的数据集
    train_dataset = SplitChatDataset("train_data.json", tokenizer)
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 训练参数 - 调整参数以适应拆分后的数据
    training_args = TrainingArguments(
        output_dir="./new_lora_model",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to=None,
        weight_decay=0.01,
        warmup_ratio=0.1,
        label_names=["labels"]
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained("./new_lora_model")
    
    print("训练完成！")

if __name__ == "__main__":
    train_model()
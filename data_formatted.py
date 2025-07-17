import json
import random

def format_data_for_training(input_file="data.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        messages = item["conversation"]
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append({
                    "role": "system",
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                formatted_messages.append({
                    "role": "user", 
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                formatted_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
            elif msg["role"] == "tool":
                formatted_messages.append({
                    "role": "tool",
                    "content": msg["content"]
                })
        
        formatted_data.append({
            "messages": formatted_messages
        })
    
    # 打乱数据
    random.shuffle(formatted_data)
    
    # 分割训练集和测试集
    train_size = int(len(formatted_data) * 1.0)
    train_data = formatted_data[:train_size]
    test_data = formatted_data[train_size:]
    
    # 保存训练数据
    with open("train_data.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    # 保存测试数据
    # with open("test_data.json", "w", encoding="utf-8") as f:
    #     json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练数据: {len(train_data)} 条")
    print(f"测试数据: {len(test_data)} 条")

if __name__ == "__main__":
    format_data_for_training()
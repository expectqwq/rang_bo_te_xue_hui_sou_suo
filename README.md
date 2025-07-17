为了让Qwen-0.5b-Instruct学会利用一个名为 "search" 的工具，本项目利用DeepSeek-R1-Distill-Qwen-7B合成了400条需要搜索和不需要搜索的工具，并通过lora微调使Qwen-0.5b-Instruct获得了一定的调用搜索能力。以下是各项目文件的作用：

1.fake_search:利用DeepSeek-R1-Distill-Qwen-7B本地部署的模拟搜索引擎。

2.data_synthesis:利用本地部署的DeepSeek-R1-Distill-Qwen-7B合成数据，直接运行即可生成数据。

3.data_formatted:修改数据格式并划分测试集与训练集。

4.new_lora_train:对Qwen-0.5b-Instruct进行lora微调，直接运行即可进行微调。

5.demo.py:运行微调后的模型，其中包含一些基本的测试用例。

6.new_lora_model:存放模型权重的地方。

经实验测试，微调过的模型能比较好地以正确格式调用网络搜索，但是产生了一定过度依赖于搜索工具的嫌疑，通过加强prompt和平衡需要搜索和不需要搜索的数据比例可以在一定程度上平衡这个缺陷，但是仍无法完全解决，初步推测是因为DeepSeek-R1-Distill-Qwen-7B也并不是一个很优秀的模型，其对是否进行搜索的判断也不够合理。


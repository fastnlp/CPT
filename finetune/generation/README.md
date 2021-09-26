# Fine-tuning CPT for Text Generation

## requirements

transformers==4.4.2

datasets==1.4.1

## data sample

adgen
```
{"article": "[[类型, 裤], [风格, 简约], [风格, 潮], [图案, 格子], [图案, 几何], [图案, 线条], [裤长, 七分裤], [裤型, 阔腿裤]]", "summarization": "这款阔腿裤，整体设计简约利落，时尚的阔腿款式带来鲜明的几何设计美感，褪去传统装束的厚重与臃肿，更具轻盈美感。 搭配七分裤长修饰出挺拔的腿部线条，气质的格纹图案不显单调，尽显女性优雅气质。 斜门襟设计潮流出众，让你时刻保持动人的女性风采。"}
```

lcsts
```
{"summarization": "可穿戴技术十大设计原则", "article": "本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代人"}
```

csl
```
{"summarization": "网络编码在实时战术数据多播中的应用", "article": "抽象了一种基于中心的战术应用场景与业务,并将网络编码技术应用于此类场景的实时数据多播业务中。在分析基于中心网络与Many-to-all业务模式特性的基础上,提出了仅在中心节点进行编码操作的传输策略以及相应的贪心算法。分析了网络编码多播策略的理论增益上界,仿真试验表明该贪心算法能够获得与理论相近的性能增益。最后的分析与仿真试验表明,在这种有中心网络的实时数据多播应用中,所提出的多播策略的实时性能要明显优于传统传输策略。"}
```

We put some data from the adgen dataset (10 train + 10 dev + 10 test) at the demo_data/adgen/ directory, which can be used to run a demo sample. 

## run

The code to run the demo sample:
```
python run_gen.py --model_path /path/to/checkpoint --dataset adgen --data_dir demo_data
```
then the training results as well as generation results will be listed at ./output/adgen/1/ .

## evaluation

We first use the BertTokenizer to process the predictions and labels into characters like follows:
```
王 旭 明 ： 现 在 的 语 文 课 至 少 有 一 半 不 该 学
彭 博 社 ： 阿 里 巴 巴 或 将 估 值 低 于 分 析 师 预 期
美 银 同 意 支 付 166 . 5 亿 美 元 和 解 mbs
凡 客 为 保 盈 利 牺 牲 如 风 达 业 务
人 大 代 表 ： 养 老 险 每 多 缴 1 年 养 老 金 应 多 发 5 %
```

After that we use [ROUGE](https://github.com/pltrdy/rouge) or [BLEU-4](https://github.com/TsinghuaAI/CPM-2-Finetune/blob/b37b07da4bf834c7a3b7e8188662df91eddb9b0a/generation_metrics.py#L89) to evaluate the generation results. The training script use ROUGE to evaluate the model after each epoch. To evaluate the generation results output by run_gen.py using BLEU-4, you can change the variables in run_bleu.py and run it:
```
python run_bleu.py
```
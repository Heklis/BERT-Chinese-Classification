# -BERT-

代码用于食品安全评论数据集，若应用于其他数据集请自行更改util.py文件。

# 环境

1. foodsafe_data放入源文件：train.csv，test_new.csv,  sample.csv
     直接从o2o比赛页面下载，无需改名
     https://www.datafountain.cn/competitions/370

2. foodsafe_data也保存结果文件：result.csv线上成绩 0.9277。

3. output是模型保存路径和验证结果保存路径。

4. 运行环境：python 3，pytorch 1.1，transformers 2.1.1。

5. 运行方式：先运行split_dataset.py， 再修改run.bat中的model_name_or_path为你自己的bert预训练模型路径，再运行 run.bat 即可。

6. GPU：1080 Ti （可通过降低序列长度和 batch size 来减弱对显存的要求）





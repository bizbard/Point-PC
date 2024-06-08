》》代码环境：遵循https://github.com/yuxumin/PoinTr环境设置
》》整个数据流的逻辑：残缺点云经过key_encoder编码，与预存储的memory_key计算相似度，返回top-3的memory_value，
memory_value经过value_encoder编码作为先验知识特征，与残缺点云特征拼接喂入decoder做最终重建。
》》代码执行顺序：
1. init_memory_keys.py --初始化memory network
2. pretrain.py --使用基于对比学习的预训练机制预训练key_encoder和value_encoder
3. train.py --训练
4. test.py --测试
5. visualization.py --可视化（需要下载mitsuba）
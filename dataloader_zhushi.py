import torch
import numpy as np

class DataLoader:
    """
    数据加载器类，用于加载特征、标签、市值和股票索引数据，并提供迭代批次和每日数据的方法。
    """

    def __init__(self, df_feature, df_label, df_market_value, df_stock_index, batch_size=800, pin_memory=True, start_index=0, device=None):
        """
        初始化数据加载器。

        参数：
        - df_feature：特征数据 DataFrame
        - df_label：标签数据 DataFrame
        - df_market_value：市值数据 Series
        - df_stock_index：股票索引数据 Series
        - batch_size：批大小，默认为800
        - pin_memory：是否将数据存储在固定内存中，默认为True
        - start_index：开始索引，默认为0
        - device：设备，默认为None
        """

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values  # 特征数据转换为 NumPy 数组
        self.df_label = df_label.values  # 标签数据转换为 NumPy 数组
        self.df_market_value = df_market_value  # 市值数据
        self.df_stock_index = df_stock_index  # 股票索引数据
        self.device = device  # 设备

        if pin_memory:  # 如果设置了 pin_memory
            self.df_feature = torch.as_tensor(self.df_feature, dtype=torch.float, device=device)  # 转换为 torch.Tensor，并移动到设备上
            self.df_label = torch.as_tensor(self.df_label, dtype=torch.float, device=device)  # 转换为 torch.Tensor，并移动到设备上
            self.df_market_value = torch.as_tensor(self.df_market_value, dtype=torch.float, device=device)  # 转换为 torch.Tensor，并移动到设备上
            self.df_stock_index = torch.as_tensor(self.df_stock_index, dtype=torch.long, device=device)  # 转换为 torch.Tensor，并移动到设备上

        self.index = df_label.index  # 获取标签数据的索引

        self.batch_size = batch_size  # 批大小
        self.pin_memory = pin_memory  # 是否将数据存储在固定内存中
        self.start_index = start_index  # 开始索引

        self.daily_count = df_label.groupby(level=0).size().values  # 每日数据点的数量
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # 每日数据点的累计索引
        self.daily_index[0] = 0  # 第一天的累计索引为0

    @property
    def batch_length(self):
        """
        获取批次的长度。

        返回：
        - 批次长度
        """

        if self.batch_size <= 0:  # 如果批大小小于等于0，则返回每日数据的长度
            return self.daily_length

        return len(self.df_label) // self.batch_size  # 否则返回批次的长度

    @property
    def daily_length(self):
        """
        获取总共的天数。

        返回：
        - 数据中的总天数
        """

        return len(self.daily_count)  # 返回数据中的总天数

    def iter_batch(self):
        """
        迭代批次数据。

        返回：
        - 批次数据迭代器
        """

        if self.batch_size <= 0:  # 如果批大小小于等于0，则迭代每日数据（随机顺序）
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))  # 创建索引数组
        np.random.shuffle(indices)  # 打乱索引顺序

        for i in range(len(indices))[::self.batch_size]:  # 每次取批大小个索引
            if len(indices) - i < self.batch_size:  # 如果剩余索引不足一个批次大小
                break
            # yield可以实现懒加载，节省内存并且提高执行速度，详细介绍见 https://zhuanlan.zhihu.com/p/142780894
            yield i, indices[i:i + self.batch_size]  # 生成器返回批次索引

    def iter_daily_shuffle(self):
        """
        迭代每日数据（随机顺序）。

        返回：
        - 每日数据迭代器（随机顺序）
        """

        indices = np.arange(len(self.daily_count))  # 创建每日索引数组
        # print("batch_length", self.daily_length)
        np.random.shuffle(indices)  # 打乱每日索引顺序

        for i in indices:  # 对于每日索引数组中的每个索引
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])  # 生成器返回每日数据的切片索引

    def iter_daily(self):
        """
        迭代每日数据。

        返回：
        - 每日数据迭代器
        """

        indices = np.arange(len(self.daily_count))  # 创建每日索引数组

        for i in indices:  # 对于每日索引数组中的每个索引
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])  # 生成器返回每日数据的切片索引

    def get(self, slc):
        """
        获取指定切片的数据。

        参数：
        - slc：切片

        返回：
        - 指定切片的特征、标签、市值、股票索引和索引数据
        """

        outs = self.df_feature[slc], self.df_label[slc][:, 0], self.df_market_value[slc], self.df_stock_index[slc]  # 获取指定切片的特征、标签、市值和股票索引数据

        if not self.pin_memory:  # 如果不将数据存储在固定内存中
            outs = tuple(torch.as_tensor(x, device=self.device) for x in outs)  # 将数据转换为 torch.Tensor，并移动到设备上

        return outs + (self.index[slc],)  # 返回指定切片的特征、标签、市值、股票索引和索引数据

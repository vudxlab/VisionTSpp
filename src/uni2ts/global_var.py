
# def _init():  # 初始化
import threading

# 创建锁对象
global lock
lock = threading.Lock()

global _global_dict
_global_dict = {"total_samples": 0, "filter_samples": 0, "dataset_count": {}, "dataset_filter": {}}

def set_value(key, value):
    # 获取锁
    with lock:
        # 定义一个全局变量
        _global_dict[key] = value

def get_value(key):
    # 获取锁：
    with lock:
        # 获得一个全局变量，不存在则提示读取对应变量失败
        try:
            return _global_dict[key]
        except:
            print('读取' + key + '失败\r\n')
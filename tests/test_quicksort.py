#!/usr/bin/env python3
"""
快速排序算法测试示例
"""

def quicksort(arr):
    """
    快速排序算法实现
    
    Args:
        arr: 待排序的列表
    
    Returns:
        排序后的列表
    """
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]  # 选择中间元素作为基准
    left = [x for x in arr if x < pivot]      # 小于基准的元素
    middle = [x for x in arr if x == pivot]   # 等于基准的元素
    right = [x for x in arr if x > pivot]     # 大于基准的元素
    
    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr, low=0, high=None):
    """
    原地快速排序算法实现（节省空间）
    
    Args:
        arr: 待排序的列表
        low: 起始索引
        high: 结束索引
    """
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分区操作，返回基准元素的正确位置
        pivot_index = partition(arr, low, high)
        
        # 递归排序基准左边和右边的子数组
        quicksort_inplace(arr, low, pivot_index - 1)
        quicksort_inplace(arr, pivot_index + 1, high)


def partition(arr, low, high):
    """
    分区函数：将数组分为小于基准和大于基准的两部分
    
    Args:
        arr: 数组
        low: 起始索引
        high: 结束索引
    
    Returns:
        基准元素的最终位置
    """
    pivot = arr[high]  # 选择最后一个元素作为基准
    i = low - 1        # 较小元素的索引
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # 交换元素
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]  # 将基准放到正确位置
    return i + 1


# 使用示例
if __name__ == "__main__":
    # 测试非原地版本
    test_arr1 = [64, 34, 25, 12, 22, 11, 90]
    print(f"原数组: {test_arr1}")
    sorted_arr1 = quicksort(test_arr1.copy())
    print(f"排序后: {sorted_arr1}")
    
    # 测试原地版本
    test_arr2 = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n原数组: {test_arr2}")
    quicksort_inplace(test_arr2)
    print(f"排序后: {test_arr2}")
    
    # 测试边界情况
    print(f"\n空数组: {quicksort([])}")
    print(f"单元素: {quicksort([42])}")
    print(f"已排序: {quicksort([1, 2, 3, 4, 5])}")
    print(f"逆序: {quicksort([5, 4, 3, 2, 1])}")
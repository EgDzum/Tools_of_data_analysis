# Дан массив размера n и длина окна k. Надо найти медиану в скользящем окне.
# Пример: nums = np.array([1,3,-1,-3,5,3,6,7]), k = 3 => ответ [1,-1,-1,3,5,6]
import heapq
import numpy as np

def find_window_median(array: np.ndarray, k: int) -> np.ndarray | None:
    res = np.zeros(len(array) - k + 1)
    array = list(array)
    for window_start in range(0, len(array) - k + 1):
        window = array[window_start:window_start+k]
        min_heap = []
        max_heap = [window[0]]
        # push small elements to max-heap and large elements to min-heap
        for ind in range(1, len(window)):
            # if input element is larger than top of max-heap,
            # we push it to max-heap
            if window[ind] <= max_heap[0]:
                # heapq does not have an interface to work with max-heap,
                # so we add an element to the end of the array and heapify it
                max_heap.append(window[ind])
                heapq._heapify_max(max_heap)
            else:
                heapq.heappush(min_heap, window[ind])

            # we normalize the number of elements in both heaps
            # to keep median on top of max-heap
            if len(max_heap) - len(min_heap) > 1:
                largest = max_heap.pop(0)
                heapq.heappush(min_heap, largest)
            if len(min_heap) > len(max_heap):
                smallest = heapq.heappop(min_heap)
                max_heap.append(smallest)
                heapq._heapify_max(max_heap)

        # median is the largest element of max-heap
        if k % 2 != 0:
            res[window_start] = max_heap[0]
        else:
            res[window_start] = (max_heap[0] + min_heap[0]) / 2
    return res

if __name__ == "__main__":
    arr = np.array([1,3,-1,-3,5,3,6,7])
    window_size = 3
    print(find_window_median(arr, window_size))

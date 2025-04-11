# -*- coding: utf-8 -*-
import numpy as np
import time
import pandas as pd


def main():
    # Start timing.
    start_time = time.time()

    # Load weight data
    weight = np.loadtxt('../result/SCN.txt')
    matching_connect_reorder = []
    image = len(weight)

    # Convert matrix to list
    origin_count = 0
    for i in range(image - 1):
        for j in range(i + 1, image):
            if weight[i, j] != 0:
                origin_count += 1
                # Record matching relationships
                matching_connect_reorder.append([i + 1, j + 1])

    matching_connect_reorder = np.array(matching_connect_reorder)

    # Load reordering data
    reorder = np.loadtxt('record__order.txt')

    # Adjust the index based on the reordered data.
    for i in range(len(matching_connect_reorder)):

        current_val = matching_connect_reorder[i, 0]
        if current_val != reorder[current_val - 1, 1]:
            matching_connect_reorder[i, 0] = reorder[current_val - 1, 1]


        current_val = matching_connect_reorder[i, 1]
        if current_val != reorder[current_val - 1, 1]:
            matching_connect_reorder[i, 1] = reorder[current_val - 1, 1]

    # 5. Save to Excel
    df = pd.DataFrame(matching_connect_reorder)
    df.to_excel('matching.xlsx', index=False, header=False)


    elapsed_time = time.time() - start_time
    print(f"Operation time consumption：{elapsed_time} s")


if __name__ == '__main__':
    main()
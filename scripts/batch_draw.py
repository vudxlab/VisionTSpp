import hashlib
import subprocess
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

showed = [
    'weighted_7_quantile_512_multi_color_9_heads',
    'weighted_7_quantile_512_multi_color_9_heads_large',
    'weighted_7_quantile_512_multi_color_9_heads_huge',
]

name_step_result = [
    [0.7054353364902101, 0.6280454580408119, 0.5927768285699287, 0.5971860721163003, 0.6104680637707507, 0.5997692458831919, 0.5818807693907567, 0.5742054018252981, 0.5719609441961033, 0.5668874314611334, 0.561],
    [0.7054353364902101, 0.6070247148741815, 0.5897042285853908, 0.6007085704731535, 0.5873750724330669, 0.5811918939734826, 0.5847496955310711, 0.5739975278338094, 0.5728430632171281, 0.5681491872752883, 0.565],
    [0.7054353364902101, 0.599966932653215, 0.5827632389401813, 0.5735675579226809, 0.5950174688844021, 0.567293547936578, 0.5681096866678684, 0.5837530148625081, 0.5648361740689309, 0.5655844815973405, 0.562],
]


showed = [x + "_missing" for x in showed]

plt.figure(figsize=(6, 8))
plt.subplot(2, 1, 1)

plt.title("Normalized MAE on Monash")
for n, y in zip(showed, name_step_result):
    x = [step*1000 for step in range(11)]
    print(x)
    print(y)
    # plt.plot(x, y, label=n, markersize=5, marker='o')
    
    # Plot the line and get its color
    line, = plt.plot(x, y, label=n, markersize=5, marker='o')
    color = line.get_color()  # Get the color of the line
    # # Add text labels near each point
    # for xi, yi in zip(x, y):
    #     plt.text(xi+0.1, yi+0.005, f'{yi:.3f}', color=color, fontsize=5,
    #              ha='right', va='bottom')  # Adjust text alignment as needed
    
plt.xlabel("Training Step")
plt.grid()
plt.legend(fontsize=7)


plt.tight_layout()
plt.savefig("result_draw.pdf")
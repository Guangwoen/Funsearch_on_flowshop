import numpy as np

from flowshop_test.utils import *
from original import neh
from evolved import evolved_neh


def main(use_evolved=False):

    filename = 'reeves13.txt'
    subdir = 'reeves'
    fs_data = load_datasets(f'/Users/cuiguangyuan/Documents/CityU/SemesterB/Artificial Intelligence/project/Funsearch_on_flowshop/data/{subdir}')[filename]
    fs_data = np.array(fs_data)

    if use_evolved:
        schedule = evolved_neh(fs_data)
    else:
        schedule = neh(fs_data)

    final_makespan = calc_makespan(schedule, fs_data)

    plot_gantt_chart(schedule, fs_data)

    return schedule, final_makespan


if __name__ == '__main__':
    schedule, final_makespan = main()

    print(f"Best schedule: {schedule}")
    print(f"Best makespan: {final_makespan}")

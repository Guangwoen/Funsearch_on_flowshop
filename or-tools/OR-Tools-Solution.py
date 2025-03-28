import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

from ortools.sat.python import cp_model


def read_cases(path):
    cases = []
    with open(path, 'r') as f:
        alldata = f.readlines()
        first_line = alldata[0].split()
        n_jobs, n_machines = int(first_line[0]), int(first_line[1])

        for i in range(1, len(alldata)):
            line = alldata[i]
            jobs_cases = []
            data = line.split()
            for d in range(0, len(data), 2):
                jobs_cases.append((int(data[d]), int(data[d+1])))
            cases.append(jobs_cases)

    return (n_jobs, n_machines), cases


def plot_gantt_chart(result_job_schedule, num_jobs, num_machines,
                     title="Flow-Shop Gantt Chart"):
    fig, ax = plt.subplots(figsize=(25, 12))

    colors = list(mcolors.TABLEAU_COLORS.values())
    if num_jobs > len(colors):
        random.seed(4487)
        colors = []
        for _ in range(num_jobs):
            colors.append(f'#{random.randint(0, 0xFFFFFF):06x}')

    for job_id, machine_id, stime, etime in result_job_schedule:
        duration = etime - stime
        rect = patches.Rectangle(
            (stime, num_machines - machine_id - 1),
            duration,
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=colors[job_id],
            alpha=0.6,
            label=f'Job-{job_id}' if machine_id == 0 else ''
        )
        ax.add_patch(rect)

        rx, ry = rect.get_xy()
        ax.text(
            rx + duration / 2,
            ry + 0.4,
            f'J{job_id}',
            ha='center',
            va='center',
            color='black',
            fontweight='light'
        )

    ax.set_xlim(0, max([endtime for _, _, _, endtime in result_job_schedule]) + 1)
    ax.set_ylim(0, num_machines)

    ax.set_yticks(np.arange(num_machines) + 0.4)
    ax.set_yticklabels([f'Machine {num_machines - i - 1}' for i in range(num_machines)])

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    handles = [patches.Patch(color=colors[i], label=f'Job {i}') for i in range(num_jobs)]
    ax.legend(handles=handles, loc='upper right', ncol=min(5, num_jobs))

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')

    plt.tight_layout()
    plt.show()


def solve_flowshop(num_jobs, num_machines, jobs_data):
    model = cp_model.CpModel()


    """Create interval variables"""
    intervals = {}
    for job_id in range(num_jobs):
        for task_id, (machine_id, duration) in enumerate(jobs_data[job_id]):
            # a job is consisted of multiple tasks

            unique_id = f'{job_id}-{machine_id}-{task_id}'
            start = model.NewIntVar(0, 10000, f's-{unique_id}')
            end = model.NewIntVar(0, 10000, f'e-{unique_id}')
            interval = model.NewIntervalVar(start, duration, end, f'interval-{unique_id}')

            # uniquely identified by job ID and machine ID
            # since a task can only be executed on a machine
            intervals[(job_id, machine_id)] = (start, end, interval)


    """Add constraints on the order"""
    for job_id in range(num_jobs):
        for task_id in range(1, num_machines):
            # (task number = machine number) in flow-shop

            prev_task_machine = jobs_data[job_id][task_id - 1][0]
            cur_task_machine = jobs_data[job_id][task_id][0]

            # start time of current task must be larger than the end time of previous task
            model.Add(intervals[(job_id, cur_task_machine)][0]
                      >= intervals[(job_id, prev_task_machine)][1])


    """Add constraints on the machine conflicts"""
    for machine_id in range(num_machines):
        machine_intervals = [
            intervals[(job_id, machine_id)][2]
            for job_id in range(num_jobs)
        ]
        model.AddNoOverlap(machine_intervals)


    """Setting Objects"""
    case_object = model.NewIntVar(0, 10000, 'makespan')
    model.AddMaxEquality(case_object, [
        intervals[(job_id, num_machines-1)][1]
        for job_id in range(num_jobs)
    ])
    model.Minimize(case_object)


    """Solving"""
    solver = cp_model.CpSolver()
    stat = solver.Solve(model)


    """Results"""
    result_schedule = []
    if stat == cp_model.OPTIMAL:
        # exists optimal solution
        print(f'Optimal Makespan: {solver.ObjectiveValue()}')
        for job_id in range(num_jobs):
            for machine_id in range(num_machines):
                start = solver.Value(intervals[(job_id, machine_id)][0])
                duration = jobs_data[job_id][machine_id][1]
                end = start + duration
                print(f'Task-{machine_id} of Job-{job_id} is scheduled on Machine-{machine_id}: {start} ~ {end}')
                result_schedule.append((job_id, machine_id, start, end))
    else:
        # No optimal solution
        print('No solution found')

    return result_schedule


if __name__ == '__main__':
    case_set_name = 'reeves'
    case_no = 20
    case_path = f'../data/{case_set_name}/{case_set_name}{case_no}.txt'

    (n_jobs, n_machines), jobs_data = read_cases(case_path)
    result_schedule = solve_flowshop(n_jobs, n_machines, jobs_data)

    plot_gantt_chart(result_schedule, n_jobs, n_machines, f'Result of case: {case_set_name}-{case_no}')

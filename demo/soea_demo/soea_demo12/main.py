# -*- coding: utf-8 -*-
"""该案例展示如何观察进化过程：用 outFunc 记录每一代的最优路线，求解后把 att48（48 城市）旅行商问题的收敛过程
保存为 GIF 动画（左图为当代最优路线，右图为最短路程随代数的变化）。

outFunc 在每一代结束时被调用，参数为算法对象与当代种群；同样的写法可以记录任何问题每一代的最优解、平均值等。
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import geatpy as ea

if __name__ == '__main__':
    problem = ea.benchmarks.TSP('att48')  # 48 座城市的旅行商问题
    history = []  # 每一代的 (代数, 最短路程, 最优路线)

    def outFunc(alg, pop):  # 每一代结束时调用
        best = np.argmin(pop.ObjV[:, 0])
        history.append((alg.currentGen, pop.ObjV[best, 0], pop.Phen[best].astype(int)))

    algorithm = ea.soea_SEGA_templet(problem,
                                     ea.Population(Encoding='P', NIND=100),
                                     MAXGEN=1000,
                                     logTras=0,
                                     outFunc=outFunc)
    res = ea.optimize(algorithm, seed=1, verbose=False, drawing=0, outputMsg=True, drawLog=False, saveFlag=False,
                      progress=True)

    # 只取路线有变化的代（再均匀抽到最多 120 帧），GIF 不会太大
    frames = [h for i, h in enumerate(history) if i == 0 or h[1] < history[i - 1][1]]
    frames = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 120)).astype(int))]
    frames.append(history[-1])  # 最后一代（之后路线不再变化）
    gens = [h[0] for h in history]
    lengths = [h[1] for h in history]

    fig, (ax_route, ax_curve) = plt.subplots(1, 2, figsize=(11, 5))
    ax_curve.plot(gens, lengths, color='lightgray')
    ax_curve.set_xlabel('Generation')
    ax_curve.set_ylabel('Shortest tour length')
    curve, = ax_curve.plot([], [], color='tab:blue')
    marker, = ax_curve.plot([], [], 'o', color='tab:red')
    ax_route.scatter(problem.places[:, 0], problem.places[:, 1], s=12, color='black', zorder=3)
    route, = ax_route.plot([], [], color='tab:blue', lw=1.2)
    ax_route.set_xticks([])
    ax_route.set_yticks([])
    ax_route.set_title(' ')  # 先占住标题的位置，tight_layout 才会给它留空间

    def draw(frame):
        gen, length, tour = frame
        xy = problem.places[np.append(tour, tour[0])]  # 回到出发城市
        route.set_data(xy[:, 0], xy[:, 1])
        upto = [i for i, g in enumerate(gens) if g <= gen]
        curve.set_data([gens[i] for i in upto], [lengths[i] for i in upto])
        marker.set_data([gen], [length])
        ax_route.set_title('Generation %d   length %.0f' % (gen, length))
        return route, curve, marker

    fig.tight_layout()
    animation = FuncAnimation(fig, draw, frames=frames + [frames[-1]] * 10, blit=False)  # 末帧停留一会儿
    animation.save('tsp_evolution.gif', writer=PillowWriter(fps=10))
    print('收敛过程已保存为 tsp_evolution.gif')
    plt.show()

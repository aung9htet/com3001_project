import numpy as np
from ecolab3.agents import Worker, Scout, Queen, Nest
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc


def run_ecolab(env, agents, Niterations=100, earlystop=True):
    """
    Run ecolab, this applies the rules to the agents and the environment. It records
    the grass array and the locations (and type) of agents in a list it returns.
    
    Arguments:
    - env = an Environment object
    - agents = a list of agents (all inherited from Agent)
    - Niterations = number of iterations to run (default = 1000)
    - earlystop = if true (default), will stop the simulation early if no agents left.
    """

    record = []
    for it in range(Niterations):
        if (it + 1) % 100 == 0: print("%5d" % (it + 1), end="\r")  # progress message

        # for each agent, apply rules (move, eat, breed)
        for agent in agents:
            agent.move(env, agents)
            agent.eat(env, agents)
            a = agent.breed(agents)
            if a is not None:
                [agents.append(newAgent) for newAgent in a]

        # removed dead agents
        agents = [a for a in agents if not a.die()]
        # Update population:
        [agent for agent in agents if type(agent) == Nest][0].updatePopulation(agents)

        # grow more grass
        env.grow()
        # Decrease pheromone levels:
        env.fadePheromones()
        # Do rainfall:
        env.causeRainFall()

        # record the grass and agent locations (and types) for later plotting & analysis
        record.append({'food': env.food.copy(), 'pheromones': env.pheromones.copy(),
                       'agents': np.array([a.summary_vector() for a in agents])})

        # stop early if we run out of rabbits and foxes
        if earlystop:
            if len(agents) == 0: break
    return record


def draw_animation(fig, record, fps=20, saveto=None):
    """
    Draw the animation for the content of record. This doesn't use the draw
    functions of the classes.
    - fig figure to draw to
    - record = the data to draw
    - fps = frames per second
    - saveto = where to save it to
    """
    # rc('animation', html='html5')
    if len(record) == 0: return None

    imF = plt.imshow(np.zeros_like(record[0]['food']), interpolation='none', aspect='auto', vmin=0, vmax=3, cmap='gray')
    ax = plt.gca()

    # foxesplot = ax.plot(np.zeros(1), np.zeros(1), 'bo', markersize=10)
    # rabbitsplot = ax.plot(np.zeros(1), np.zeros(1), 'yx', markersize=10, mew=3)
    workerPlot = ax.plot(np.zeros(1), np.zeros(1), 'bx', markersize=10, mew=3)
    scoutPlot = ax.plot(np.zeros(1), np.zeros(1), 'rx', markersize=10, mew=3)
    queenPlot = ax.plot(np.zeros(1), np.zeros(1), 'gx', markersize=10, mew=3)
    nestPlot = ax.plot(np.zeros(1), np.zeros(1), 'yo', markersize=7.5)

    def animate_func(i):
        imF.set_array(record[i]['food'])
        ags = record[i]['agents']
        if len(ags) == 0:
            workerPlot[0].set_data([], [])
            scoutPlot[0].set_data([], [])
            queenPlot[0].set_data([], [])
            nestPlot[0].set_data([], [])
            return
        # Plot workers:
        coords = np.array([[x, y] for x, y, z in ags if z == 0])
        workerPlot[0].set_data(coords[:, 1], coords[:, 0])
        # Plot scouts:
        coords = np.array([[x, y] for x, y, z in ags if z == 1])
        scoutPlot[0].set_data(coords[:, 1], coords[:, 0])
        # Plot Queens:
        coords = np.array([[x, y] for x, y, z in ags if z == 2])
        queenPlot[0].set_data(coords[:, 1], coords[:, 0])
        # Plot Nest:
        coords = np.array([[x, y] for x, y, z in ags if z == 3])
        nestPlot[0].set_data(coords[:, 1], coords[:, 0])
        # return [im]#,rabbits,foxes]

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=len(record),
        interval=1000 / fps, repeat=False  # in ms
    )
    if saveto is not None: anim.save(saveto, fps=fps, extra_args=['-vcodec', 'libx264'])
    from IPython.display import HTML
    return HTML(anim.to_jshtml())


def get_agent_counts(record):
    """
    Returns the number of workers, scouts, queens, nests, & food in an N x 5 numpy array
    the three columns are (Worker, Scout, Queen, Nest, Food).
    """
    counts = []
    for r in record:
        ags = r['agents']
        if len(ags) == 0:
            nW = 0
            nS = 0
            nQ = 0
            nN = 0
        else:
            nW = np.sum(ags[:, -1] == 0)
            nS = np.sum(ags[:, -1] == 1)
            nQ = np.sum(ags[:, -1] == 1)
            nN = np.sum(ags[:, -1] == 1)
        nF = np.sum(r['food'])
        counts.append([nW, nS, nQ, nN, nF])
    counts = np.array(counts)
    return counts

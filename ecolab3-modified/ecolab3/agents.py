import numpy as np
import matplotlib.pyplot as plt


# Helper functions to calculate distances
def calcdistsqr(v):
    """Get euclidean distance^2 of v"""
    return np.sum(v ** 2)


def calcdist(v):
    """Get euclidean distance of v"""
    return np.sqrt(np.sum(v ** 2))


def getNest(agents):
    """
    Only works assuming there's one nest per simulation.
    ----------- NEEDS TO BE CHANGED IF NOT -----------
    """
    return [agent for agent in agents if type(agent) == Nest][0]


class Agent:
    """
    Base class for all types of agent
    """

    def __init__(self, position, age, food, speed, lastBreed, maxAge, starveThresh, inNest=True,
                 breedFood=None, breedFreq=None, isCarryingFood=False, returningToNest=False):
        """
        breedFood = quantity of food required to breed
        breedFreq = how many iterations between breed attempts
        maxAge = max age of agent
        isCarryingFood = whether or not the agent is carrying food
        returningToNest = if the agent is returning to the nest
        starveThresh = how low food levels have to be before trying to eat
        inNest = whether or not the agent is inside the nest
        age = age of agent in iterations
        food = how much food the agent has 'inside' (0=empty, 1=full)
        position = x,y position of the agent
        speed = how fast it can move (tiles/iteration)
        lastBreed = how long ago it last reproduced (iterations)
        """
        self.breedFood = breedFood
        self.breedFreq = breedFreq
        self.maxAge = maxAge
        self.isCarryingFood = isCarryingFood
        self.returningToNest = returningToNest
        self.starveThresh = starveThresh
        self.inNest = inNest
        self.food = food
        self.age = age
        self.position = position
        self.speed = speed
        self.lastBreed = lastBreed

    def breed(self):
        """
        This will either return None, or a new agent object
        Having breed freq/food be none means never breed
        """
        if self.breedFreq is None or self.breedFood is None:
            return None
        new_agent = None
        if (self.lastBreed > self.breedFreq) and (self.food > self.breedFood):
            self.lastBreed = -1
            new_agent = type(self)(self.position, 0, self.food / 2, self.speed, 10)
            self.food = self.food / 2
        self.age += 1
        self.lastBreed += 1
        return new_agent

    def move(self, env, agents):
        pass  # to implement by child class

    def tryMove(self, newPosition, env):
        if env.check_position(newPosition):
            self.position = newPosition
        # ensures it's in the environment and rounds to nearest cell
        # env.fix_position(self.position)

    def attemptMoveToTarget(self, targetPos, env):
        relativeTargetPos = targetPos - self.position
        if calcdistsqr(relativeTargetPos) < self.speed ** 2:
            self.tryMove(targetPos, env)
        else:
            vect = relativeTargetPos / calcdist(relativeTargetPos)
            self.tryMove(self.position + vect * self.speed, env)

    def eat(self, env, agents):
        pass  # to implement by child class

    def workerEat(self, env, agents, pheromoneRange, vision):
        """
        Eat method for both scouts and workers. Placed in parent class to reduce code duplication. Cannot be the
        default method due to the Queen's eat method being different.
        - Go to the closest source of food:
                - Nest
                - Food deposit
                - Food being carried (In that order)
        Eat if below starving threshold, or if in nest - Nest provides faster refill of food levels.
        Ants can carry more food than any individual can eat, giving a large one-time boost to food levels at the
        expense of the food being carried.
        """
        # Eat if on top of food:
        if env.get_food(self.position) > 0:
            env.reduce_food(self.position)
            self.food += 1
        # If below starving threshold:
        if (self.food < self.starveThresh):
            """
            Find the nearest source of food and eat(increase food level)
            """
            # Get nest position
            nestPos = getNest(agents).getPos()
            nestDist = calcdistsqr(nestPos)
            # If can see nest:
            if (nestDist < pheromoneRange ** 2):
                self.attemptMoveToTarget(nestPos, env)
            # Can't sense nest, look for closest food deposit:
            else:
                foodPos = env.get_loc_of_food(self.position, vision)
                if foodPos is not None:
                    self.attemptMoveToTarget(foodPos, env)
            # Last resort - eat the food currently being carried:
            if self.isCarryingFood:
                self.isCarryingFood = False
                self.returningToNest = False
                self.food += 2
        # Replenish food quickly if in nest:
        if self.inNest:
            self.food += 3
        # If not in the nest, reduce food:
        else:
            self.food -= 1

    def die(self):
        """
        Returns true if it needs to expire, either due to:
            - no food left
            - old age
        """
        if self.food <= 0:
            return True
        if self.age > self.maxAge:
            return True
        return False

    def getPos(self):
        return self.position

    def summary_vector(self):
        """
        Returns a list of the location (x,y) and a 0=fox, 1=rabbit, e.g.
        [3,4,1] means a rabbit at (3,4).
        """
        return [self.position[0], self.position[1], type(self) == Rabbit]


class Nest(Agent):
    """
    Nest class, tbc
    """
    def __init__(self, position, age=None, food=10, speed=1, lastBreed=0, starveThresh=3):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh)
        self.eaten = False


class Worker(Agent):
    vision = 3  # How many squares the agent can see
    pheromoneRange = 10  # How many squares the agent can see specifically pheromones
    isCarryingFood = False  # Whether or not the agent is carrying a unit of food
    breedFreq = None
    breedFood = None
    maxAge = 20

    def __init__(self, position, age=None, food=10, speed=1, lastBreed=0, starveThresh=3):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh)
        self.eaten = False

    def move(self, env, agents):
        """
        Worker Ant movement:
        - If out of nest:
        - If near pheromones,
            - Follow trail to food
            - Buff up the trail while on it (unless no food is left)
        - If near food,
            - Pick up food
            - Return to nest
        """
        # If an ant in the nest is in a state to forage:
        if (self.inNest and self.food > self.starveThresh * 3):
            """
            Check for nearest pheromones, and begin to follow trail. If not, do nothing.
            """

    def eat(self, env, agents):
        self.workerEat(env, agents, self.pheromoneRange, self.vision)


class Scout(Agent):
    vision = 10  # How many squares the agent can see
    pheromoneRange = 6  # How many squares the agent can see specifically pheromones
    isCarryingFood = False  # Whether or not the agent is carrying a unit of food
    breedFreq = None
    breedFood = None
    maxAge = 20

    def __init__(self, position, age=None, food=20, speed=3, lastBreed=0, starveThresh=2):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh)
        self.eaten = False

    def move(self, env, agents):
        """
        Search for the closest food deposit:
        - Move around somewhat randomly until getting in range of food
        - When at food:
            - Pick up some food
            - Travel in a line back to the nest,
            - Leave a pheromone trail along the way
        """
        # If returning to nest:
        if self.returningToNest:
            # Get nest pos:
            nestPos = getNest(agents).getPos()
            # Get direction of travel:
            dy = float(nestPos[1]) - self.position[1]
            dx = float(nestPos[0]) - self.position[0]
            direction = np.tan(dy / dx)

            # Move towards the nest:
            delta = np.round(np.array([np.cos(direction), np.sin(direction)]) * self.speed)
            self.tryMove(self.position + delta, env)

            # Place pheromones over the path:
            env.increase_pheromone(self.position, 0.3)
            env.increase_pheromone(self.position + delta, 0.3)
        else:
            # If in nest & in a state to search for food:
            if (self.inNest and self.food > self.starveThresh * 3):
                foodPos = env.get_loc_of_food(self.position, self.vision)
                if foodPos is not None:
                    self.attemptMoveToTarget(foodPos, env)
                else:
                    # no food in range, pick a random direction...
                    d = np.random.rand() * 2 * np.pi  # pick a random direction
                    delta = np.round(np.array([np.cos(d), np.sin(d)]) * self.speed)

                    self.tryMove(self.position + delta, env)
            # If on top of food:
            if (env.get_food(self.position) > 0):
                # Pick up food:
                self.isCarryingFood = True
                env.reduce_food(self.position)
                # Return to nest, leaving pheromones:
                self.returningToNest = True


    def eat(self, env, agents):
        self.workerEat(env, agents, self.pheromoneRange, self.vision)


class Rabbit(Agent):
    # These are the same for all rabbits.
    vision = 5  # how far it can see around current tile
    breedFreq = 10  # how many iterations have to elapse between reproduction events
    breedFood = 10  # how much food has to be eaten to allow reproduction
    maxAge = 40  # how long do they live

    def __init__(self, position, age=None, food=10, speed=1, lastBreed=0):
        """
        A Rabbit agent. Arguments:
        age = age of agent in iterations (default is a random value between 0 and maxAge)
        food = how much food the agent has 'inside' (0=empty, 1=full), default = 10
        position = x,y position of the agent (required)
        speed = how fast it can move (tiles/iteration) (default=1)
        lastBreed = how long ago it last reproduced (iterations) (default=0)
        """
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed)
        self.eaten = False

    def move(self, env):
        """
        rabbit movement:
         - if current cell has no food...
            - will move towards cells with more food
         - DOESN'T move away from nearby foxes
        """
        if env.get_food(self.position) == 0:
            food_position = env.get_loc_of_grass(self.position,
                                                 self.vision)  # get the x,y location of nearby food (if any)
            if food_position is not None:
                relative_food_position = food_position - self.position
                if calcdistsqr(
                        relative_food_position) < self.speed ** 2:  # if distance to the food < how far we can go, then
                    self.tryMove(food_position, env)

                else:
                    vect = relative_food_position / calcdist(relative_food_position)
                    self.tryMove(self.position + vect * self.speed, env)
            else:
                # no food in range, pick a random direction...
                d = np.random.rand() * 2 * np.pi  # pick a random direction
                delta = np.round(np.array([np.cos(d), np.sin(d)]) * self.speed)

                self.tryMove(self.position + delta, env)

    def eat(self, env, agents):
        """
         - will eat if there's food at location
         - otherwise food goes down by 1.
        """
        if env.get_food(self.position) > 0:
            env.reduce_food(self.position)
            self.food += 1
        else:
            self.food -= 1

    #    def draw(self):
    #        plt.plot(self.position[0],self.position[1],'yx',mew=3)

    def die(self):
        """
        Returns true if it needs to expire, either due to:
         - no food left
         - old age
         - being eaten
        """
        if self.food <= 0:
            return True
        if self.age > self.maxAge:
            return True
        if self.eaten:
            return True
        return False


class Fox(Agent):
    # These are the same for all foxes.
    vision = 7  # how far it can see around current tile
    breedFreq = 30  # how many iterations have to elapse between reproduction events
    breedFood = 20  # how much food has to be eaten to allow reproduction
    maxAge = 80  # how long do they live

    def __init__(self, position, age=None, food=10, speed=5, lastBreed=0):
        """
        A Fox agent. Arguments:
        age = age of agent in iterations (default is random age between 0 and maxAge)
        food = how much food the agent has 'inside' (0=empty, 1=full) (default=10)
        position = x,y position of the agent (required)
        speed = how fast it can move (tiles/iteration) (default=5)
        lastBreed = how long ago it last reproduced (iterations) (default=0)
        """
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed)

    def get_nearby_rabbit(self, position, vision, agents):
        """
        helper function, given the list of agents, find the nearest rabbit, if within 'vision', else None.
        """
        # distances to dead rabbits and foxes set to infinity.
        sqrdistances = np.sum((np.array(
            [a.position if (type(a) == Rabbit) and (not a.die()) else np.array([-np.inf, -np.inf]) for a in
             agents]) - position) ** 2, 1)
        idx = np.argmin(sqrdistances)
        if sqrdistances[idx] < vision ** 2:
            return agents[idx]
        else:
            return None

    def eat(self, env, agents):
        """     
        Eat nearby rabbit, with a probability that drops by distance
        """
        near_rabbit = self.get_nearby_rabbit(self.position, self.vision,
                                             agents)  # get the x,y location of nearby rabbit (if any)
        if near_rabbit is not None:
            relative_food_position = near_rabbit.position - self.position
            dist = calcdist(relative_food_position)
            if dist < self.speed:  # if distance to the food < how far we can go, then
                # probability that fox will kill rabbit is ratio of speed to distance
                kill_prob = 1 - (dist / self.speed)
                if kill_prob > np.random.rand():
                    self.tryMove(near_rabbit.position, env)
                    near_rabbit.eaten = True
                    self.food += 2  # near_rabbit.food/2

    def move(self, env):
        """
        Foxes just move randomly (but also move during call to self.eat eating to catch rabbit).
        """
        d = np.random.rand() * 2 * np.pi  # pick a random direction
        delta = np.round(np.array([np.cos(d), np.sin(d)]) * self.speed)
        self.tryMove(self.position + delta, env)

    def die(self):
        """
        Returns true if it needs to expire, due to either:
         - no food
         - old age
        """
        if self.food <= 0:
            return True
        if self.age > self.maxAge:
            return True
        return False

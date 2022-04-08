import numpy as np
from ecolab3.environment import Environment


# Helper functions to calculate distances
def calcdistsqr(v):
    """Get euclidean distance^2 of v"""
    return np.sum(v ** 2)


def calcdist(v):
    """Get euclidean distance of v"""
    return np.sqrt(np.sum(v ** 2))


def getNest(agents):
    """
    Returns the nest agent from the agents array:
    """
    nest = [agent for agent in agents if type(agent) == Nest][0]
    return nest


class Agent:
    """
    Base class for all types of agent
    """

    def __init__(self, position, age, food, speed, lastBreed, maxAge, starveThresh, vision=1, pheromoneRange=1,
                 inNest=True, breedFood=None, breedFreq=None, isCarryingFood=False, isReturningToNest=False):
        """
        breedFood = quantity of food required to breed
        breedFreq = how many iterations between breed attempts
        maxAge = max age of agent
        isCarryingFood = whether or not the agent is carrying food
        isReturningToNest = if the agent is returning to the nest
        starveThresh = how low food levels have to be before trying to eat
        vision = how far the agent can see
        pheromoneRange = how far the agent can sense pheromones
        inNest = whether or not the agent is inside the nest
        age = age of agent in iterations
        food = quantity of food within the agent
        position = x,y position of the agent
        speed = how fast it can move (tiles/iteration)
        lastBreed = how long ago it last reproduced (iterations)
        """
        self.vision = vision
        self.pheromoneRange = pheromoneRange
        self.breedFood = breedFood
        self.breedFreq = breedFreq
        self.maxAge = maxAge
        self.isCarryingFood = isCarryingFood
        self.isReturningToNest = isReturningToNest
        self.starveThresh = starveThresh
        self.inNest = inNest
        self.food = food
        self.age = age
        self.position = position
        self.speed = speed
        self.lastBreed = lastBreed

    def breed(self, agents):
        pass

    def move(self, env, agents):
        pass  # to implement by child class

    def followTrail(self, env, sense, agents):
        """
        Get the location of the pheromones that are either as far from the nest or as close
        to the nest as possible.
        """
        searchSquare = env.generateSearchSquare(self.getPos(), sense, False)

        if np.all(searchSquare <= 0):
            return None  # no target instances found

        positions = np.array([np.zeros(len(searchSquare)) for _ in range(len(searchSquare))])
        # Get the center index of the searchSquare (ie the agent's position):
        center = (len(searchSquare) + 1) / 2

        # Each position is translated from local to global coords - relative to the agent's position:
        # Positions are only calculated for squares that have >0 pheromones,
        # as otherwise there's no point in checking them.
        for i in range(len(searchSquare)):
            for j in range(len(searchSquare[i])):
                if searchSquare[i][j] > 0:
                    diff = np.array((center - j) * (-1), center - i)
                    positions[i][j] = np.array(self.getPos()[0] + diff[0], self.getPos()[1] + diff[1])
                else:
                    positions[i][j] = np.nan

        # Get distance for each position:
        distances = np.array([[calcdistsqr(pos - getNest(agents).getPos())
                               for pos in row if pos != np.nan] for row in positions])

        # If the agent is returning to the nest, get the pheromone closest to the nest.
        # Otherwise, get the pheromones furthest from the nest
        if self.isReturningToNest:
            # Get the minimum distance:
            targetDistance = np.nanargmin(distances)
        else:
            # Get the maximum distance:
            targetDistance = np.nanargmax(distances)

        # Return the position([x,y]) of the pheromones closest/furthest from the nest:
        return positions.flatten()[targetDistance]

    def tryMove(self, newPosition, env):
        if env.check_position(newPosition):
            self.position = newPosition

    def attemptMoveToTarget(self, targetPos, env):
        relativeTargetPos = targetPos - self.getPos()
        if calcdistsqr(relativeTargetPos) < self.speed ** 2:
            self.tryMove(targetPos, env)
        else:
            vect = relativeTargetPos / calcdist(relativeTargetPos)
            newPos = self.getPos() + vect * self.speed, env
            self.tryMove(newPos, env)

    def eat(self, env, agents):
        pass  # to implement by child class

    def leavePheromones(self, start, finish, env, amount=0.1):
        """
        Using list comprehension, leave pheromones over the path travelled by an ant.
        """
        [[env.increase_pheromone([i, j], amount) for j in range(round(start[1]), round(finish[1]))] for i in
         range(round(start[0]), round(finish[0]))]

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
        if self.food >= self.starveThresh * 5:
            # Eat if on top of food:
            if env.get_food(self.getPos()) > 0:
                env.reduce_food(self.getPos())
                self.food += 1
            # If below starving threshold:
            if (self.food < self.starveThresh):
                # Get nest position
                nestPos = getNest(agents).getPos()
                nestDist = calcdistsqr(nestPos)
                # If can see nest:
                if (nestDist < pheromoneRange ** 2):
                    self.attemptMoveToTarget(nestPos, env)
                # Can't sense nest, look for closest food deposit:
                else:
                    foodPos = env.get_loc_of_food(self.getPos(), vision)
                    if foodPos is not None:
                        self.attemptMoveToTarget(foodPos, env)
                    else:
                        # Last resort - eat the food currently being carried:
                        if self.isCarryingFood:
                            self.isCarryingFood = False
                            self.isReturningToNest = False
                            self.food += 2
            # Replenish food quickly if in nest:
            if self.inNest:
                self.food += 3
                getNest(agents).reduceFood()
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
        return np.array([x for x in self.position if type(x) != Environment]).flatten()

    def summary_vector(self):
        """
        Returns a list of the location (x,y) and a 0=worker, 1=scout, 2=queen, 3=nest.
        [3,4,1] means a scout at (3,4).
        """
        ants = [Worker, Scout, Queen, Nest]
        return [self.getPos()[0], self.getPos()[1], ants.index(type(self))]


class Nest(Agent):
    """
    Nest class
    The class is the one and only and will manage the population of queens, worker ants and scout ants as such
    """
    maxAge = 100000
    population = [0, 0, 0]  # Workers, Scouts, Queens

    def __init__(self, position, age=1, food=80, speed=1, lastBreed=0, starveThresh=20):
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh)

    def addPopulation(self, ant):
        """
        Add ants to the nest 
        """
        if type(ant) == Worker:
            self.population[0] += 1
        elif type(ant) == Scout:
            self.population[1] += 1
        elif type(ant) == Queen:
            self.population[2] += 1

    def depositFood(self, amount=2):
        self.food += amount

    def reduceFood(self, amount=1):
        self.food -= amount

    def updatePopulation(self, agents):
        for agent in agents:
            if type(agent) == Worker:
                self.population[0] += 1
            elif type(agent) == Scout:
                self.population[1] += 1
            elif type(agent) == Queen:
                self.population[2] += 1


class Queen(Agent):
    """
    Required for calculating the population of the nest
    """
    breedFood = 40
    breedFreq = 50
    maxAge = 350

    def __init__(self, position, age=None, food=10, maxfood=100, speed=1, lastBreed=0, starveThresh=5):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh, breedFood=self.breedFood,
                         inNest=True)
        self.maxfood = maxfood

    def eat(self, env, agents):
        if self.food < self.maxfood:
            # Get food from nest:
            getNest(agents).reduceFood(2)
            # Eat:
            self.food += 3

    def breed(self, agents):
        """
            This will either return None, or a new agent object
            Having breed freq/food be none means never breed
        """
        if self.breedFreq is None or self.breedFood is None:
            return None

        newAgents = None

        if (self.lastBreed > self.breedFreq) and (self.food > self.breedFood):
            self.lastBreed = -1
            newAgents = self.createNewAgents(agents)
            self.food = self.food / 2

        return newAgents

    def createNewAgents(self, agents):
        """
        Using the information from the Nest, decide which type of ant the new agent should be.
        """
        # Number of new agents to create:
        numAgents = 20
        newAgents = np.empty(numAgents, dtype=Agent)
        # The portion of the Queen's food to give to the new agents:
        foodForAgent = (self.food / 2) / numAgents
        # How much food is stored in the nest in comparison to the nest's starve threshold:
        nest = getNest(agents)
        foodRatio = nest.food / nest.starveThresh

        for i in range(len(newAgents)):
            # Decide which type of agent to create:
            agentCaste = np.random.randint(0, 100)
            if agentCaste < 75:
                newAgents[i] = type(Worker)(self.getPos(), 0, foodForAgent, 1, 0, 20, 3, 4, 10)
            else:
                newAgents[i] = type(Scout)(self.getPos(), 0, foodForAgent, 3, 0, 20, 2, 10, 5)

        # Check if new queen needs to be created:
        if self.age > (2 * self.maxAge) / 3:
            # Create new queen:
            newAgents[0] = type(Queen)(self.getPos(), 0, foodForAgent, self.speed, -1, self.maxAge, self.starveThresh,
                                       breedFood=self.breedFood)

        return newAgents


class Worker(Agent):
    vision = 4  # How many squares the agent can see
    pheromoneRange = 10  # How many squares the agent can see specifically pheromones
    isCarryingFood = False  # Whether or not the agent is carrying a unit of food
    maxAge = 20

    def __init__(self, position, age=None, food=10, speed=1, lastBreed=0, starveThresh=3):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh, self.vision,
                         self.pheromoneRange)

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
        # If in nest:
        if np.array_equal(self.getPos(), getNest(agents).getPos()) or self.inNest:
            self.inNest = True
            if self.isCarryingFood:
                getNest(agents).depositFood()
                self.isCarryingFood = False
            return
        # If on top of food:
        if env.get_food(self.getPos()) > 0:
            # Pick up food:
            self.isCarryingFood = True
            env.reduce_food(self.getPos())
            # Return to nest, leaving pheromones:
            self.isReturningToNest = True
            return
        # If in nest and fed enough:
        if self.inNest and self.food > self.starveThresh * 2:
            trail = self.followTrail(env, self.pheromoneRange, agents)
            if trail is not None:
                self.attemptMoveToTarget(trail, env)
        else:
            # Check if agent can see food:
            foodPos = env.get_loc_of_food(self.getPos(), self.vision)
            if foodPos is not None:
                startPos = self.getPos()
                # Attempt to move to target, leaving pheromones along the way:
                self.attemptMoveToTarget(foodPos, env)
                self.leavePheromones(startPos, self.getPos(), env)
            else:
                # If not, continue to follow the trail:
                trail = self.followTrail(env, self.pheromoneRange, agents)
                if trail is not None:
                    startPos = self.getPos()
                    self.attemptMoveToTarget(trail, env)
                    if self.isReturningToNest:
                        # Leave pheromones over the path:
                        self.leavePheromones(startPos, self.getPos(), env)
                else:
                    # Try to return to nest if lost:
                    self.isReturningToNest = True

    def eat(self, env, agents):
        self.workerEat(env, agents, self.pheromoneRange, self.vision)


class Scout(Agent):
    vision = 10  # How many squares the agent can see
    pheromoneRange = 5  # How many squares the agent can see specifically pheromones
    isCarryingFood = False  # Whether or not the agent is carrying a unit of food
    breedFreq = None
    breedFood = None
    maxAge = 20

    def __init__(self, position, age=None, food=20, speed=3, lastBreed=0, starveThresh=2):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed, self.maxAge, starveThresh, self.vision,
                         self.pheromoneRange)
        self.eaten = False

    def lookForFood(self, env):
        foodPos = env.get_loc_of_food(self.getPos(), self.vision)
        if foodPos is not None:
            self.attemptMoveToTarget(foodPos, env)
        else:
            d = np.random.rand() * 2 * np.pi  # pick a random direction
            delta = np.round(np.array([np.cos(d), np.sin(d)]) * self.speed)
            self.attemptMoveToTarget([self.getPos() + delta], env)

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
        if self.isReturningToNest:
            self.returnToNest(env, agents)
            print("4")
        else:
            # If in nest & in a state to search for food:
            if self.inNest and self.food > self.starveThresh * 2:
                self.lookForFood(env)
                self.inNest = False
                print("1")
            # If on top of food:
            if env.get_food(self.getPos()) > 0:
                # Pick up food:
                self.isCarryingFood = True
                env.reduce_food(self.getPos())
                # Return to nest, leaving pheromones:
                self.isReturningToNest = True
                print("2")
            else:
                self.lookForFood(env)
                print("3")

    def returnToNest(self, env, agents):
        """
        Move towards the nest. If at the nest, deposit food.
        """
        # If returning to nest:
        if self.isReturningToNest:
            # Get nest pos:
            nestPos = getNest(agents).getPos()

            if np.array_equal(self.getPos(), nestPos):
                # Deposit Food:
                self.inNest = True
                self.isReturningToNest = False
                if self.isCarryingFood:
                    getNest(agents).depositFood()
                    self.isCarryingFood = False
            else:
                startPos = self.getPos()
                self.attemptMoveToTarget(nestPos, env)
                # Get direction of travel:
                """
                dy = float(nestPos[1]) - self.getPos()[1]
                dx = float(nestPos[0]) - self.getPos()[0]
                direction = np.tan(dy / dx)

                # Move towards the nest:
                delta = np.round(np.array([np.cos(direction), np.sin(direction)]) * self.speed)
                startPos = self.getPos()
                self.tryMove(self.getPos() + delta, env)
                """

                # Place pheromones over the path:
                self.leavePheromones(startPos, self.getPos(), env, 0.2)

    def eat(self, env, agents):
        self.workerEat(env, agents, self.pheromoneRange, self.vision)


"""
class Rabbit(Agent):
    # These are the same for all rabbits.
    vision = 5  # how far it can see around current tile
    breedFreq = 10  # how many iterations have to elapse between reproduction events
    breedFood = 10  # how much food has to be eaten to allow reproduction
    maxAge = 40  # how long do they live

    def __init__(self, position, age=None, food=10, speed=1, lastBreed=0):
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed)
        self.eaten = False

    def move(self, env):
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
        if env.get_food(self.position) > 0:
            env.reduce_food(self.position)
            self.food += 1
        else:
            self.food -= 1

    #    def draw(self):
    #        plt.plot(self.position[0],self.position[1],'yx',mew=3)

    def die(self):
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
        if age is None:
            age = np.random.randint(self.maxAge)
        super().__init__(position, age, food, speed, lastBreed)

    def get_nearby_rabbit(self, position, vision, agents):
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
        d = np.random.rand() * 2 * np.pi  # pick a random direction
        delta = np.round(np.array([np.cos(d), np.sin(d)]) * self.speed)
        self.tryMove(self.position + delta, env)

    def die(self):
        if self.food <= 0:
            return True
        if self.age > self.maxAge:
            return True
        return False
"""

"""
Code adapted from
- Author: Mike Croucher
- Environment.py
- Source code for ecolab3 python version
- Source: https://github.com/lionfish0/ecolab3.git
"""

from re import S
import numpy as np
import matplotlib.pyplot as plt


# helper function
def argmax_2darray(a):
    """
    Returns the maximum location in a n-d array
    """
    return np.unravel_index(a.argmax(), a.shape)


class Environment:
    def __init__(self, shape=None, startfood=40, maxfood=80, maxfoodperunit=20, droprate=10, amountPerDrop=10,
                 max_percentage=0, rain_intensity=0, percentage_dry=0):
        """
        Create the environment
        Parameters:
         - shape = shape of the environment
         - startfood = initial amount of food to be started on the map rounded to nearest 10
         - maxfood = maximum amount of food allowed in the whole map rounded to nearest 10
         - maxfoodperunit = maximum amount of food per unit allowed rounded to nearest 10
         - droprate = number of tiles which get extra food each iteration(the droprate can only be lesser than the one decribed here)
         - max_percentage = percentage of wetlands to be able to exist at most
         - rain_intensity = intensity of rain which will also decide how much wetlands will be formed
         - percentage_dry = probability of wetlands turning to dry lands
        """
        if shape is None:
            shape = [80, 80]
        self.rain_intensity = rain_intensity  # set rain intensity to randomized the wet land formed and pheromone to
        # reduce depending on the rain intensity
        self.maxfood = round(maxfood / 10) * 10  # maximum it can grow to, should be in increments of 10
        self.droprate = droprate  # how many new items of food added per step
        self.amountPerDrop = amountPerDrop  # quantity of food per drop
        self.shape = shape  # shape of the environment
        self.maxfoodperunit = round(maxfoodperunit / 10) * 10  # decide the max food size per unit
        self.startfood = round(startfood / 10) * 10  # starting point of food, should be in increments of 10
        self.food = np.zeros(self.shape)  # will be randomised later on map
        self.pheromones = np.zeros(self.shape)  # set all pheromone trail as zero initially
        self.waterLevels = np.zeros(self.shape)  # 0 being completely dry, 1 being completely waterlogged
        self.rainFall = 0  # Intensity of rainfall, 0 being none, 1 being heavy
        self.max_percentage = max_percentage  # decide the max number of wet lands to be produced
        self.percentage_dry = percentage_dry  # decide to change wet land to dry
        self.status_reduce_pher = True #decide to reduce pheromone or not

    def causeRainFall(self, intensity=None, chanceOfRain=None):
        """
        Run all methods related to causing rainfall.
        """
        if chanceOfRain is None:
            chanceOfRain = 70
        # Set rain intensity for current iteration:
        self.generateRainIntensity(intensity, chanceOfRain)
        # Cause rain to effect environment:
        self.calculateWaterLevels()

    def generateRainIntensity(self, val=None, chanceOfRain=70):
        """
        Generate value for rain intensity.
        """
        if np.random.randint(0, 100) < chanceOfRain:
            if val is None:
                # Generate new intensity:
                newIntensity = np.random.randint(0, 100) / 100
            else:
                newIntensity = val
        else:
            newIntensity = 0

        self.rainFall = newIntensity

    def getRainIntensity(self):
        """
        Return value for rainfall intensity.
        """
        return self.rainFall

    def getLevelIncrease(self, currentLevel):
        """
        Using the value of rainfall (+/- 10%), calculate the increase in water level it would cause.
        -- At rain intensity 'x', the increase would be 'x / 5' --
        """
        percentChange = np.random.randint(-10, +10) / 100
        increase = (self.rainFall + (self.rainFall * percentChange)) / 5

        newLevel = increase + currentLevel

        if newLevel > 1:
            return 1
        else:
            return newLevel

    def getPheromoneDecrease(self, currentLevel):
        """
        Using the value of rainfall, calculate the effect it would have on reducing the pheromone levels.
        """
        if currentLevel != 0:
            decrease = self.rainFall / 5

            newLevel = currentLevel - decrease

            if newLevel >= 0:
                return newLevel

        return 0

    def calculateWaterLevels(self):
        """
        Using the current rainfall & water levels, calculate the water levels of each position after
        an iteration of rain.
        """
        decayValue = 0.2

        if self.rainFall != 0:
            # If it is raining:
            self.waterLevels = [[self.getLevelIncrease(lvl) for lvl in row] for row in self.waterLevels]
            # Also calculate the effects on pheromones:
            if self.status_reduce_pher:
                self.pheromones = [[self.getPheromoneDecrease(lvl) for lvl in row] for row in self.pheromones]
        else:
            # If it isn't raining:
            self.waterLevels = [[lvl - decayValue if lvl > decayValue else 0 for lvl in row]
                                for row in self.waterLevels]

    def get_pheromone(self, position):
        """
        Returns the pheromone amount at that point
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        return self.pheromones[position[0]][position[1]]

    def reduce_pheromone(self, position, amount):
        """
        Reduce the amount of pheromone at position by amount with 0 being the least it can be reduced to
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        lvl = self.pheromones[position[0]][position[1]]
        if lvl <= 0 or lvl - amount <= 0:
            self.pheromones[position[0]][position[1]] = 0
        else:
            self.pheromones[position[0]][position[1]] -= amount

    def increase_pheromone(self, position, amount):
        """
        Increase the amount of pheromone at position by amount with 1 being the most it can be increased to
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        lvl = self.pheromones[position[0]][position[1]]
        if lvl >= 1 or lvl + amount >= 1:
            self.pheromones[position[0]][position[1]] = 1
        else:
            self.pheromones[position[0]][position[1]] += amount

    def get_food(self, position):
        """
        Returns the amount of food at position
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        return self.food[position[0]][position[1]]

    def reduce_food(self, position, amount=1):
        """
        Reduce the amount of food at position by amount
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        lvl = self.food[position[0]][position[1]]
        if lvl <= 0 or lvl - amount <= 0:
            self.food[position[0]][position[1]] = 0
        else:
            self.food[position[0]][position[1]] -= amount

    def increase_food(self, position, amount):
        """
        Increase the amount of food at position
        """
        position = np.array([int(x) for x in position if type(x) != Environment]).flatten()
        lvl = self.food[position[0]][position[1]]
        if lvl >= self.maxfoodperunit or lvl + amount >= self.maxfoodperunit:
            self.food[position[0]][position[1]] = self.maxfoodperunit
        else:
            self.food[position[0]][position[1]] += amount

    def get_loc_of_food(self, position, sense):
        position = np.array([x for x in position if type(x) != Environment]).flatten()
        return self.get_loc_of_target(position, sense, True)

    def get_loc_of_pheromone(self, position, sense):
        position = np.array([x for x in position if type(x) != Environment]).flatten()
        return self.get_loc_of_target(position, sense, False)

    def generateSearchSquare(self, position, vision, isFood=True):
        """
        Get the search square around the position of the agent and return it.
        """
        # we temporarily build a new datastructure to look for the largest food in with a
        # strip/boundary around the edge of zero food. This feels like the simplest code
        # to solve the edge case problem, but is quite a slow solution.
        if isFood:
            target = self.food
        else:
            target = np.array(self.pheromones * 10)
        position = np.array([x for x in position if type(x) != Environment]).flatten()
        boundary = 30
        pos = np.array(position) + boundary
        targetWithBoundary = np.zeros(np.array(target.shape) + boundary * 2)
        targetWithBoundary[boundary:-boundary, boundary:-boundary] = target
        # we search just a circle within 'vision' tiles of 'pos' (these two commands build that search square)
        searchSquare = targetWithBoundary[int(np.floor(pos[0]) - vision):int(np.floor(pos[0]) + vision + 1),
                       int(np.floor(pos[1]) - vision):int(np.floor(pos[1]) + vision + 1)]
        searchSquare[(np.arange(-vision, vision + 1)[:, None] ** 2 + np.arange(-vision, vision + 1)[None,
                                                                     :] ** 2) > vision ** 2] = -1
        return searchSquare

    def get_loc_of_target(self, position, vision, isFood=True):
        """
        This finds the location of the cell with the maximum amount of food near 'pos',
        within a circle of 'vision' size.
        For example env.get_dir_of_food(np.array([3,3]),2)
        if two or more cells have the same food then it will select between them randomly.
        """
        position = np.array([x for x in position if type(x) != Environment]).flatten()
        searchSquare = self.generateSearchSquare(position, vision, isFood)

        if np.all(searchSquare <= 0):
            return None  # no target instances found
        return argmax_2darray(searchSquare + 0.01 * np.random.rand(vision * 2 + 1, vision * 2 + 1)) + position - vision

    def randomise_food_initially(self):
        """
        Puts food randomly on the map with size that is random
        """
        self.grow(round(self.startfood / 10))

    def check_position(self, position):
        """
        Returns whether the position is within the environment
        """
        position = np.array([x for x in position if type(x) != Environment]).flatten()

        if position[0] < 0:
            return False
        if position[1] < 0:
            return False
        if position[0] > self.shape[0] - 1:
            return False
        if position[1] > self.shape[1] - 1:
            return False

        # this adds a 'wall' across the environment...
        # if (position[1]>5) and (position[0]>self.shape[0]/2-3) and (position[0]<self.shape[0]/2+3): return False
        return True

    def get_center(self):
        """
        Returns the middle position of the environment
        """
        position = [int(np.floor(self.shape[1] / 2)), int(np.floor(self.shape[0] / 2))]
        position = np.array([x for x in position if type(x) != Environment]).flatten()
        return position

    def get_random_location(self):
        """
        Returns a random location in the environment.
        """
        return np.random.randint([0, 0], np.array(self.shape) - [5,5])

        # if we have a more complicated environment shape, use this instead to place new food in valid location...
        # p = np.array([-10,-10])
        # while not self.check_position(p):
        #    p = np.random.randint([0,0],self.shape)
        # return p

    def grow(self, numDrops=None):
        """
        Adds more food (random locations) 
         - amount added controlled by self.droprate, self.maxfood and self.maxfoodperunit
        """
        if numDrops is None:
            numDrops = self.droprate
        for i in range(0, numDrops):
            totalFood = np.sum(self.food)
            # If the maximum amount of food in the environment is reached:
            if totalFood >= self.maxfood or (totalFood + self.amountPerDrop) >= self.maxfood:
                # Break out of the loop, stopping the increase:
                break
            # Get (randomised) location to add food:
            location = self.get_random_location()
            # Make sure it is a valid drop location:
            while (self.food[location[0], location[1]] + self.amountPerDrop) >= self.maxfoodperunit:
                location = self.get_random_location()
            # Drop the food at the location:
            self.food[location[0], location[1]] += self.amountPerDrop

    def fadePheromones(self):
        """
        Reduce the concentration of pheromones over time:
        """
        decay = 0
        self.pheromones = [[lvl - decay if lvl > decay else 0 for lvl in row] for row in self.pheromones]

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
    def __init__(self, shape=[40, 40], startfood=1, maxfood=3, droprate=10, max_percentage=0, rain_intensity=0,
                 percentage_dry=0):
        """
        Create the environment
        Parameters:
         - shape = shape of the environment
         - startfood = initial amount of food
         - maxfood = maximum amount of food allowed in each tile
         - droprate = number of tiles which get extra grass each iteration
         - max_percentage = percentage of wetlands to be able to exist at most
         - rain_intensity = intensity of rain which will also decide how much wetlands will be formed
         - percentage_dry = probability of wetlands turning to dry lands
        """
        self.rain_intensity = rain_intensity  # set rain intensity to randomized the wet land formed and pheromone to reduce depending on the rain intensity
        self.maxfood = maxfood  # maximum it can grow to
        self.droprate = droprate  # how many new items of food added per step
        self.shape = shape  # shape of the environment
        self.food = np.full(self.shape, startfood)  # 2*np.trunc(np.random.rand(*self.shape)*2)+2 #initial food
        self.pheromone = np.zeros(self.shape)  # set all pheromone trail as zero initially
        self.env_status = np.zeros(self.shape)  # 0 being dry land and 1 being wet land, Initialised all as wet land
        self.max_percentage = max_percentage  # decide the max number of wet lands to be produced
        self.percentage_dry = percentage_dry  # decide to change wet land to dry

    def get_rain_intensity(self):
        """
        Returns rain intensity
        """
        return self.rain_intensity

    def set_rain_intensity(self, intensity):
        """
        Set rain intensity
        """
        self.rain_intensity = intensity

    def change_env_rain(self):
        """
        Change the wet lands randomly depending on the intensity of the rain
        """
        intensity = self.get_rain_intensity
        row = len(self.env_status)
        column = len(self.env_status[0])  # column is going to be same for every row
        number_of_blocks = row * column
        max_n_wet = intensity * self.max_percentage * number_of_blocks  # decide max number of wet land blocks
        # count number of wet blocks currently
        counter = 0
        for x in self.env_status:
            counter += x.count(1)
        # generate wet lands if it is not at max limit
        n_generate_wet = 0
        if (counter < max_n_wet):
            n_generate_wet = np.random.randint(max_n_wet - counter)
            for i in range(n_generate_wet):
                valid_change = False
                while (valid_change == False):
                    n_row = np.random.randint(row)
                    n_column = np.random.randint(column)
                    if self.env_status[n_row, n_column] == 0:
                        valid_change = True
                self.set_env_rain[n_row, n_column]

    def change_env_dry(self):
        """
        Change wet lands to dry lands with a default percentage
        """
        for row in self.env_status:
            for current_unit in row:
                if current_unit == 1:
                    change_dry = np.random.rand()
                    if change_dry <= self.percentage_dry:
                        current_unit = 0  # changed to dry

    def get_env_status(self, position):
        """
        Returns the environment status
        """
        return self.env_status[int(position[0]), int(position[1])]

    def change_env_status(self, position):
        """
        Change the status of land
        """
        if (self.env_status[int(position[0]), int(position[1])] == 1):
            self.env_status[int(position[0]), int(position[1])] = 0
        else:
            self.env_status[int(position[0]), int(position[1])] = 1

    def set_env_rain(self, position):
        """
        Set status of land to rain
        """
        self.env_status[int(position[0]), int(position[1])] = 1

    def set_env_dry(self, position):
        """
        Set status of land to dry
        """
        self.env_status[int(position[0]), int(position[1])] = 0

    def get_pheromone(self, position):
        """
        Returns the pheromone amount at that point
        """
        return self.pheromone[int(position[0]), int(position[1])]

    def reduce_pheromone(self, position, amount):
        """
        Reduce the amount of pheromone at position by amount with 0 being the least it can be reduced to
        """
        if (self.pheromone[int(position[0]), int(position[1])] == 0):
            self.pheromone[int(position[0]), int(position[1])] = 0
        elif ((self.pheromone[int(position[0]), int(position[1])] - amount) < 0):
            self.pheromone[int(position[0]), int(position[1])] = 0
        else:
            self.pheromone[int(position[0]), int(position[1])] -= amount

    def increase_pheromone(self, position, amount):
        """
        Increase the amount of pheromone at position by amount with 1 being the most it can be increased to
        """
        if (self.pheromone[int(position[0]), int(position[1])] == 1):
            self.pheromone[int(position[0]), int(position[1])] = 1
        elif ((self.pheromone[int(position[0]), int(position[1])] + amount) > 1):
            self.pheromone[int(position[0]), int(position[1])] = 1
        else:
            self.pheromone[int(position[0]), int(position[1])] += amount

    def get_food(self, position):
        """
        Returns the amount of food at position
        """
        return self.food[int(position[0]), int(position[1])]

    def reduce_food(self, position, amount=1):
        """
        Reduce the amount of food at position by amount
        (note, doesn't check this doesn't go negative)
        """
        if self.get_food(position) > 0:
            self.food[int(position[0]), int(position[1])] -= amount

    def get_loc_of_food(self, position, vision):
        """
        This finds the location of the cell with the maximum amount of food near 'pos',
        within a circle of 'vision' size.
        For example env.get_dir_of_food(np.array([3,3]),2)
        if two or more cells have the same food then it will select between them randomly.
        """

        # we temporarily build a new datastructure to look for the largest food in with a
        # strip/boundary around the edge of zero food. This feels like the simplest code
        # to solve the edge case problem, but is quite a slow solution.
        boundary = 10
        pos = position + boundary
        foodwithboundary = np.zeros(np.array(self.food.shape) + boundary * 2)
        foodwithboundary[boundary:-boundary, boundary:-boundary] = self.food
        # we search just a circle within 'vision' tiles of 'pos' (these two commands build that search square)
        searchsquare = foodwithboundary[int(pos[0] - vision):int(pos[0] + vision + 1),
                       int(pos[1] - vision):int(pos[1] + vision + 1)]
        searchsquare[(np.arange(-vision, vision + 1)[:, None] ** 2 + np.arange(-vision, vision + 1)[None,
                                                                     :] ** 2) > vision ** 2] = -1
        # this returns the location of that maximum food (with randomness added to equally weight same-food cells)
        if np.all(searchsquare <= 0): return None  # no food found
        return argmax_2darray(searchsquare + 0.01 * np.random.rand(vision * 2 + 1, vision * 2 + 1)) + position - vision

    def check_position(self, position):
        """
        Returns whether the position is within the environment
        """
        position[:] = np.round(position)
        if position[0] < 0: return False
        if position[1] < 0: return False
        if position[0] > self.shape[0] - 1: return False
        if position[1] > self.shape[1] - 1: return False

        # this adds a 'wall' across the environment...
        # if (position[1]>5) and (position[0]>self.shape[0]/2-3) and (position[0]<self.shape[0]/2+3): return False
        return True

    def get_random_location(self):
        """
        Returns a random location in the environment.
        """
        return np.random.randint([0, 0], self.shape)

        # if we have a more complicated environment shape, use this instead to place new food in valid location...
        # p = np.array([-10,-10])
        # while not self.check_position(p):
        #    p = np.random.randint([0,0],self.shape)
        # return p

    def grow(self):
        """
        Adds more food (random locations) 
         - amount added controlled by self.growrate
        """
        for it in range(self.droprate):
            loc = self.get_random_location()
            if self.food[loc[0], loc[1]] < self.maxfood:
                self.food[loc[0], loc[1]] += 1

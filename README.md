# COM3001 - Modelling & Simulation of Natural Systems
## Group Project - Group 15

Group project for Modelling & Simulation of Natural Systems - Simulation (using an Agent-Based Model) of the effects of heavy and indefinate rainfall on a colony of ants typically found in a rainforest. Using a Python implementation of Dawn Walker's [Ecolab3](https://github.com/lionfish0/ecolab3).

> Contributors:
> - Aung Htet
> - Cameron Leech-Thomson
> - Henry Wilson
> - Huayi Pan

## Running the System

To run the system, open the Demo.ipynb file, and run each code segment. To change the intensity and chance of rain, edit the line `record = run_ecolab(env,agents, intensity=X, chanceOfRain=Y)`, and replace X with a rain intensity value between 0 & 1, and replace Y with a percentage chance of rain between 0 & 100. To get randomised results, you can remove the `intensity` and `chanceOfRain` declarations completely. You could also replace X & Y with `None` to achieve the same result. 

## Code

The code has been created to simulate the foraging methods of ants when in a rainforest, and how it would effect their ability to navigate the environment with their heavy reliance on pheromones. While the model itself has fallen short of its original plan, there are still some noticible emergent behaviours. The main methods & agents of the system can be found below:

### Agents

The agent parent class contains the methods that will be used across all agent types, which include:
- Breed() - Have the agent reproduce a version of itself at the cost of some food. In the case of the Queen (the only ant in a colony capable of repoducing), it will create either a Worker Ant or a Scout Ant agent normally, but will also create another Queen if the current Queen has passed 2/3 of its lifespan.
- Move() - Move the agent to its destination, or if the destination cannot be reached, move as close to the destination as possible. Both the Scout Ant and the Worker Ant have their own versions of this method. The Scout Ant will walk around the environment in a somewhat random fashion, until finding a deposit of food, in which case it will return to the nest, leaving behind a trail of pheromones. The Worker Ant will remain in the Nest until a pheromone trail has been established, in which case it will begin to follow the trail to the food, and bring some back to the nest, leaving it's own pheromones in the trail too.
- TryMove() - Check whether or not the agent's destination is valid, and if so, move to the destination.
- AttemptMoveToTarget() - The actual movement part of the `move()` method. Take the agent's target position, and check whether it can reach it in one iteration, if so, call `tryMove()`. Otherwise, get as close as possible to the destination as it can in one iteration.
- Eat() - Eat food to replenish the agent's energy. The Worker Ant and Scout Ant have the same method, while the Queen's differs slightly. The Queen is always in the nest, and therefore always has as much food as it needs, so it will always try to replenish its energy from the nest's stores. For the Worker & Scout, they will only eat if their current energy is less than 4 times their starving threshold. If so, they will find food and eat food in the following order:
    1. Eat food from the area they are standing on.
    2. If they are not in the immediate vicinity of a food deposit, travel to whichever is closer - The nest or a food deposit. In the case of the nest, the agents energy will replenish faster than getting food from any other source.
    3. If they will not survive the journey, but are carrying food. Eat the food being carried, then return to the deposit to get more.
- Die() - Checks if the agent is about to starve to death or die of old age. If so, return True, else return False. The nest will always return False.
- GetPos() - Return the `[x, y]` coordinate of the agent.
- Sumary_vector() - Return the agent's current location & type (0=Worker, 1=Scout, 2=Queen, 3=Nest).

Each agent also has it's own specialised methods to allow them to operate efficiently, they are as follows:

#### Nest
- DepositFood() - Add food to the nest. Called when an agent returns while carrying food.
- ReduceFood() - Remove food from the nest. Called when an agent eats food from the nest's stores.
- GetFood() - Return the quantity of food stored in the nest. Used for documenting.
- UpdatePopulation() - Using a list of all active agents, update the population list (`[Workers, Scouts, Queens]`).

#### Queen
- CreateNewAgent() - When breeding, decide which agent should be made per birth, and how many. The food required is split evenly amongst all new agents. There is a 3:1 ratio of Workers to Scouts, and so there is a 75% chance that the new agent will be a Worker Ant. As well as this, if a new Queen is required, one will be born alongside the new group of agents.

#### Worker
- FollowTrail() - Find the closest or furthest set of pheromones to the nest in vision, and follow them either towards or away from the nest. The direction depends on the isReturningToNest Boolean value as part of the agents. If isReturningToNest is true, the agent will travel to the pheromones closest to the nest, whereas if it is true, they will follow the pheromones furthest from the nest.

#### Scout
- LookForFood() - Search for food nearby. If there is a food deposit in the vision of the agent, travel to it. If not, move around randomly until a food deposit is found.
- ReturnToNest() - Scout Ants have an excellent sense of direction as well as a good memory. Travel in the direction of the nest until arriving at it.

### Environment
The environment object is the where the data about food, pheromones, rain, water levels, etc. is all stored. A lot of it's functions are of the type get, reduce, increase.
- CauseRainFall() - Runs all of the methods related to causing rainfall and water level increases (`generateRainIntensity()`, `calculateWaterLevels()`, `getPheromoneDecrease()`, `getLevelIncrease()`). There are parameters for the intensity of the rain and the chance of rain per iteration, to allow the results to be reproducable.
- GenerateRainIntensity() - Randomly select a number between 0-100, if it is less than the chance of rain, cause rainfall. In the case of rainfall, a random intensity of rainfall is generated (if not provided as mentioned above) between 0-1, to 3dp. If there is no rainfall, set intensity to 0.
- GetRainIntensity() - Return the value of the rain intensity.
- GetLevelIncrease() - Using the current water level of a coordinate, calculate the new water level from the rain intensity. The new level is calculated by `(intensity ± 10%) / 5 + the current water level`.
- GetPheromoneDecrease() - As the rainfall will wash away the pheromones left behind, calculate the new pheromone levels after a rainfall. It is calculated by `current pheromone level - intensity / 5`.
- CalculateWaterLevels() - Using `getLevelIncrease()`, apply the level increases to each coordinate in the environment. If there is no rain, have the water levels dry up by 0.2 per iteration.
- GetPheromone() - Get the pheromone level at specified position.
- ReducePheromone() - Reduce the pheromone level at specified position.
- IncreasePheromone() - Increase the pheromone level at specified position.
- GetFood() - Get the food level at specified position.
- ReduceFood() - Reduce the food level at specified position.
- IncreaseFood() - Increase the food level at specified position.
- GetLocationOfFood() - Using the vision component of the agent, find all locations around the agent where there is food.
- GetLocationOfPheromone() - Using the vision component of the agent, find all locations around the agent where there is pheromones.
- GetLocationOfTarget() - The broader version of the two above functions. Allowing the same method to be used to find both food or pheromones.
- GenerateSearchSquare() - The search area used for `getLocationOfTarget()`.
- RandomiseFoodInitally() - Using the `grow()` method (see below), randomly place the starting food quantity in drops of 10 across the environment.
- CheckPosition() - Checks whether or not the position provided is valid within the environment - bigger than 0, smaller than the maximum size.
- GetCenter() - Returns the center coordinate of the environment.
- GetRandomLocation() - Returns a random position in the environment.
- Grow() - Adds more food at random locations in the environment, only drops food when below the maximum amount of food in the environment, and will only drop a certain amount per iteration - this can all be controlled through the parameters in the Environment `init` method.
- FadePheromones() - Reduce the concentration of pheromones over time.

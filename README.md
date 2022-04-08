# COM3001 - Modelling & Simulation of Natural Systems
## Group Project - Group 15

Group project for Modelling & Simulation of Natural Systems - Simulation (using an Agent-Based Model) of the effects of heavy and indefinate rainfall on a colony of ants typically found in a rainforest. Using a Python implementation of Dawn Walker's [Ecolab3](https://github.com/lionfish0/ecolab3).

> Contributors:
> - Aung Htet
> - Cameron Leech-Thomson
> - Henry Wilson
> - Huayi Pan

## Code

The code has been created to simulate the foraging methods of ants when in a rainforest, and how it would effect their ability to navigate the environment with their heavy reliance on pheromones. While the model itself has fallen short of its original plan, there are still some noticible emergent behaviours. The main methods & agents of the system can be found below:

### Agents
The agent parent class contains the methods that will be used across all agent types, which include:
- Breed() - Have the agent reproduce a version of itself at the cost of some food. In the case of the Queen (the only ant in a colony capable of repoducing), it will create either a Worker Ant or a Scout Ant agent normally, but will also create another Queen if the current Queen has passed 2/3 of its lifespan.
- Move() - Move the agent to its destination, or if the destination cannot be reached, move as close to the destination as possible. Both the Scout Ant and the Worker Ant have their own versions of this method. The Scout Ant will walk around the environment in a somewhat random fashion, until finding a deposit of food, in which case it will return to the nest, leaving behind a trail of pheromones. The Worker Ant will remain in the Nest until a pheromone trail has been established, in which case it will begin to follow the trail to the food, and bring some back to the nest, leaving it's own pheromones in the trail too.
- TryMove() - Check whether or not the agent's destination is valid, and if so, move to the destination.
- AttemptMoveToTarget() - The actual movement part of the Move() method. Take the agent's target position, and check whether it can reach it in one iteration, if so, call TryMove(). Otherwise, get as close as possible to the destination as it can in one iteration.
- Eat() - Eat food to replenish the agent's energy. The Worker Ant and Scout Ant have the same method, while the Queen's differs slightly. The Queen is always in the nest, and therefore always has as much food as it needs, so it will always try to replenish its energy from the nest's stores. For the Worker & Scout, they will only eat if their current energy is less than 4 times their starving threshold. If so, they will find food and eat food in the following order:
    1. Eat food from the area they are standing on.
    2. If they are not in the immediate vicinity of a food deposit, travel to whichever is closer - The nest or a food deposit. In the case of the nest, the agents energy will replenish faster than getting food from any other source.
    3. If they will not survive the journey, but are carrying food. Eat the food being carried, then return to the deposit to get more.
- Die() - Checks if the agent is about to starve to death or die of old age. If so, return True, else return False. The nest will always return False.
- GetPos() - Return the [x, y] coordinate of the agent.
- Sumary_vector() - Return the agent's current location & type (0=Worker, 1=Scout, 2=Queen, 3=Nest).

Each agent also has it's own specialised methods to allow them to operate efficiently, they are as follows:

#### Nest
- DepositFood() - Add food to the nest. Called when an agent returns while carrying food.
- ReduceFood() - Remove food from the nest. Called when an agent eats food from the nest's stores.
- GetFood() - Return the quantity of food stored in the nest. Used for documenting.
- UpdatePopulation() - Using a list of all active agents, update the population list ([Workers, Scouts, Queens]).

#### Queen
- CreateNewAgent() - When breeding, decide which agent should be made per birth, and how many. The food required is split evenly amongst all new agents. There is a 3:1 ratio of Workers to Scouts, and so there is a 75% chance that the new agent will be a Worker Ant. As well as this, if a new Queen is required, one will be born alongside the new group of agents.

#### Worker
- FollowTrail() - Find the closest or furthest set of pheromones to the nest in vision, and follow them either towards or away from the nest. The direction depends on the isReturningToNest Boolean value as part of the agents. If isReturningToNest is true, the agent will travel to the pheromones closest to the nest, whereas if it is true, they will follow the pheromones furthest from the nest.

#### Scout
- LookForFood() - Search for food nearby. If there is a food deposit in the vision of the agent, travel to it. If not, move around randomly until a food deposit is found.
- ReturnToNest() - Scout Ants have an excellent sense of direction as well as a good memory. Travel in the direction of the nest until arriving at it.

#Arthur's bar problem 

The El Farol bar problem is a game theory theory problem
where, in particular, every thursday night, all of the people
want to go to the El Farol Bar.Since, the capacity of the bar
is small and it’s no fun to go there if it’s too crowded. So, the
preferences of the population can be described as follows:

• If less than 60 % of the population go to the bar, they’ll
all have a better time than if they stayed at home.

• If more than 60 % of the population go to the bar, they’ll
all have a worse time than if they stayed at home.

So, the problem is that everyone has to decide at the same
time whether they will go to the bar or not. They cannot
wait and see how many others go on a particular Thursday
before deciding to go themselves on that Thursday.
I here implement one variant of this bar problem as following. 
There are N players, each picking one out of K nights every week. Each agent picks the
night randomly and based on the attendance at each night
of K nights, the associated system reward (G) is calculated.
The attendance at each particular night is therefore used to
estimate the local reward of each of the agent attending that
night. Each possible move for the agent is the attendance
profile of K-dimensional vector, with each the night corre-
sponding to each agents is ’1’ and rest are ’0’. For instance,
if an agent i chooses to attend the night 3 out of k available
nights, the k-dimensional vector becomes 0, 0, 1, 0,....., 0.

Here are few assumptions based on which the problems
stated above are solved, as following.

• The agent’s action is deterministic, i.e when the agent
takes an action (chooses a night), that action takes it
with 100% probability in that particular night.

• The process is a Markov decision process, i.e the
knowledge of the current state is sufficient to character-
ize the future.

The goal here to analyze the performance of multiple
rewards used for training the system to maximize the perfor-
mance of the system.

I train the system of agents to choose the actions based on the local, global and difference rewards
and further, discuss how the system performance is changed
by each of these, and how the system performance deviates
from the optimal value (optimal attendance specified by the
bar), with each of these rewards. Further, I discuss all these
parameters with the simulation results of two specific cases,
each with different numbers of agents (N), nights available in
a week (k) and the optimal attendance (b) at each night. 

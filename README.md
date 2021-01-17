# Frozen-Lake-Reinforcement-Learning-Algorithms
Marco Anselmi, Junhui Liu,
Henry Munachi Ndubuaku
School of Electronic Engineering and Computer Science, Queen Mary University of London, UK
Abstract
The Frozen Lake Environment is a board environment with four types of tiles Start, Frozen Lake, Hole and End, in which an agent can move either horizontally or vertically. This report summarises the development of an Agent capable of traversing the Frozen Lake Environment. It also includes a literature review that analyses previous work done in the area. The techniques which have been used to obtain the optimal policy for the Agent are Policy Iteration and Value Iteration in a Tabular Model-Based Reinforcement Learning approach, Sarsa and Q-learning in a Tabular Model-Free Reinforcement Learning approach and lastly Linear Sarsa and Linear Q-learning in a Non-Tabular Model-Free Reinforcement Learning approach. Policy Evaluation will be used to evaluate the value functions found through Sarsa and Q-learning, with the most optimal policy computed using Policy Iteration.
1

Contents
1 Introduction 4
2 Literature Review 4
3 Benchmark 5
4 Background 6
4.1 FrozenLake......................................... 6
4.2 DynamicProgramming................................... 7
4.3 Tabularmodel-basedreinforcementlearning....................... 7
4.3.1 Policyevaluation:.................................. 7 4.3.2 PolicyImprovement: ................................ 8 4.3.3 PolicyIteration: .................................. 8 4.3.4 ValueIteration ................................... 8
4.4 Tabularmodel-freereinforcementlearning ........................ 8 4.4.1 SarsaControl.................................... 9 4.4.2 Q-learningcontrol ................................. 10
4.5 Non-tabularmodel-freereinforcementlearning...................... 11 4.5.1 Sarsacontrolusinglinearfunctionapproximation: . . . . . . . . . . . . . . . 11 4.5.2 Q-Learning control using linear function approximation: . . . . . . . . . . . . 11
5 Techniques Implemented 11
5.1 FrozenLakeEnvironment ................................. 11
5.2 TabularModel-BasedModel................................ 11 5.2.1 PolicyEvaluation.................................. 11 5.2.2 PolicyImprovement ................................ 11 5.2.3 PolicyIteration................................... 12 5.2.4 ValueIteration ................................... 12
5.3 TabularModel-FreeModel................................. 12 5.3.1 SarsaControl.................................... 12 5.3.2 SarsaCompare ................................... 13 5.3.3 Q-learningControl ................................. 14 5.3.4 Q-learningCompare ................................ 14
5.4 Non-TabularModel-FreeModel.............................. 15 5.4.1 LinearSarsa..................................... 15 5.4.2 LinearQ-learning.................................. 16
5.5 MainFunction ....................................... 17
6 Experimental Study 17
6.1 Tabularmodelforthebiglake(CorrespondingtoQuestion2) . . . . . . . . . . . . . 17
6.2 Sarsa and Q-learning for the small lake (Corresponding to Question 3) . . . . . . . . 18 6.2.1 Sarsacontrolwithcomparison........................... 18 6.2.2 Q-learningcontrolwithcomparison........................ 20
6.3 Sarsa and Q-learning for the big lake(Corresponding to Question 5) . . . . . . . . . . 22 6.3.1 Sarsacontrolforthebiglake ........................... 22 6.3.2 Q-learningcontrolforthebiglake ........................ 22
2

7 Discussion 24 8 Conclusion and Future Work 25 References 25
List of Figures
1 ReinforcementLearningmodel .............................. 5
2 ResultsoftheSarsaalgorithm............................... 6
3 SarsaControlFlowchart .................................. 9
4 Q-learningControlFlowchart ............................... 10
5 ResultsoftheSarsaalgorithm............................... 19
6 MatrixmultiplicationtogettheQ-Values ........................ 24
List of Tables
1 PolicyIterationandValueIterationResults ....................... 18
2 SarsaExperimentResults ................................. 19
3 Q-learningExperimentResults .............................. 21
4 SarsawithbiglakeExperimentResults ......................... 22
5 Q-learningwithbiglakeExperimentResults....................... 23
3

1 Introduction
Reinforcement Learning (RL) technology has been around since the early days of cybernetics, and it has developed from a combination of subjects such as control theory, statistics and psychology. It was not until the lat 80s and early 90s, however, that Reinforcement Learning started to become of interest to the Machine Learning (ML) and Artificial Intelligence (AI) communities [1]. Its promise of being able to model an agent only through a system of rewards and punishments, without having any knowledge regarding how the task is to be achieved, is what made it gather the interest of those communities. Furthermore, in the last decades, with the introduction of Deep Reinforcement Learning, interest in Reinforcement Learning has been increasing. This paper, however, will dive into the exploration of normal Reinforcement Learning. Reinforcement Learning can be divided into two groups based on the approach taken for finding the best model for the agent. The first approach is the model-based approach. In this methodology, the agent is given the model of the environment, therefore, the agent is capable of using the properties of the environment to find the optimal strategy to reach its goal. The second approach is the model-free approach, which uses an agent that has no knowledge regarding the environment, and only knows which actions it is allowed to take. In order for an agent to learn, it has to interact with the environment a multitude of times, and improve only using the feedback it receives from the environment. Both of the approaches have their advantages and disadvantages. The first approach has the advantage of needing little time to find an optimal strategy, however, it requires a complete model of the environment to do so, and in many real-life situations, it is not possible to have a complete model of the environment and how it behaves. The second approach, while it has no need to know anything about the environment, it has the disadvantage that it is required to interact with the environment many times before reaching a good strategy [6], thus making it very inefficient in the case where the environment has many states.
The rest of this paper will provide an overview and comparison of two algorithms for each of the approaches, which are Policy Iteration and Value Iteration for the model-based approach, and Sarsa and Q-learning for the model-free approach. Section 2, provides a summary of work that has already been done in the subject, in the form of a review of articles published prior to this paper. Section 3 explains how the framework works and how the agent interact with it. Sections 4 and 5 give an explanation of how the four algorithms work and how they were implemented. Section 6 and 7 lay out the exploration and comparison done on the algorithms, and provides the results and analysis of the results obtained through experimentation. Lastly, section 8, provides a conclusion to the paper and a consideration of what work could be done in the future on this subject.
2 Literature Review
There has been a lot of research done in the area of Reinforcement Learning, especially in recent years with the development of models which use Deep Reinforcement Learning, such as AlphaGo[5] and AlphaZero[7], which have learned how to play games such as Go. However, the majority of research focuses on using Deep Reinforcement Learning algorithms, although for a good reason, as they perform a lot better compared to normal Reinforcement Learning algorithms, thus, there hasn’t been as much recent research in general RL.
Research in the techniques presented in this paper, has been done extensively even before the in- troduction of Deep Reinforcement Learning. For model-based techniques, for example, Lagoudakis and Parr in [3] proposed a method of combining value-function approzimation with linear architec- tures and approximate policy iteration. However, due to the limitations of model-based algorithms,
4

researchers have mainly focused on model-free algorithms. Research done in this area, covers both reviews of the current state of the subject and new work being done.
Kaelbling, Littman and Moore in [1], present a summary of the history of the field, and of work which was current at the time of writing. Baird and Moore in [2], propose a new equation called VAPS, which can be instantiated to generate a wide range of new algorithms. Similarly Gao et al in [6], proposed a new algorithm called Normalized Actor-Critic (NAC), which uses a unified loss function to process both off-line demonstration data and on-line experience based on the underlying maximum entropy reinforcement learning framework.
3 Benchmark
Reinforcement Learning is the study of the creation of an agent that can be considered intelligent. A useful definition of intelligence that can be used to evaluate an agent was given by Shane Legg in the paper Machine Super Intelligence[4], after gathering definitions from other experts, he summarized the essence of intelligence as "Intelligence measures an agent’s ability to achieve goals in a wide range of environments". Using this definition of intelligence, it is possible to say that for an agent to be intelligent, the agent needs to have goals, and be able to perform actions in the environment so as to reach the goal. Thus, Reinforcement Learning works with a framework in which there exists an agent and an environment, as the agent interacts with the environment it receives a feedback/reward, the goal of the agent is to maximise the reward it gets from the environment. The basic model for Reinforcement Learning is shown in Figure 1. On each interaction that the agent
Figure 1: Reinforcement Learning model
has with the environment, the agent receives information regarding the current state, s, as input, then on the basis of the current step, it chooses an action, a, as output. The environment then takes the action chosen by the agent as input, and it produces a new state, s+1, and a reward, r+1, for having taken that action from that state and having moved into the new state.
The objective of the agent is to learn a policy, π : S × A → [0,1], which returns the probability of taking an action in a certain state, such that it maximises the discounted sum of rewards it receives from the environment. If the environment has Markov properties, meaning that a state summarizes the entire past with all that is relevant for decision making, the interaction between agent and environment is known as a Markov Decision Process (MDP). This, allows us to define a joint conditional probability over states and rewards, P(St+1 = s’, Rt+1 = r’ | st,at,rt,...,r1,s0,a0) = P(St+1 = s’, Rt+1 = r’ | st,at), known as the one-step dynamics. The one-step dynamics, is the basis
 5

forthefunctionsthatdefinethebehaviouroftheenvironment,Pass′ andRass′.Pass′ istheprobability
of transitioning from state s to state s’ given an action a, and it is given by Pass′ = P(St+1 = s′ |
St =s,At =a)=􏰀r′P(St+1 =s′,Rt+1 =r′ |St =s,At =a). WhereasRass′ istheexpected
reward on transitioning from state s to state s’ given the action a, and it is given by Rass′ = E[Rt+1 ′ ′1′′′
=r|St=s,At=a,St+1=s]=Pa 􏰀r′rP(St+1=s,Rt+1=r|St=s,At=a).Inthe ss′
 case of model-based algorithms, these functions are accessible by the algorithm and can be used directly to identify the most optimal policy. However, in the case of model-free algorithms, they don’t have access to the functions, thus, they can only optimize a policy through a trial-and-error methodology, and use the instantaneous rewards obtained from the environment.
4 Background
4.1 Frozen Lake
The frozen lake environment comprises a nxn grid representing a matrix of states, 4x4 for the small frozen lake (Fig. 2a) and 8x8 for the big frozen lake (Fig. 2b).
State variants: start (grey), frozen lake (light blue), hole (dark blue), and goal (white).
(a) Small Frozen Lake (b) Big Frozen Lake
Figure 2: Results of the Sarsa algorithm
Actions: Up, left, down, or right. Any action that moves the agent outside the grid leaves the state unchanged.
Slip: With a probability of 0.1, the environment ignores the desired direction and moves one tile in a random direction.
Reward: The agent receives reward (1) upon taking an action at the goal state. But zero in every other state.
Absorbing state: Any action taken at the goal state or in a hole moves the agent into the absorbing state. Every action taken at the absorbing state leads to the absorbing state, which also does not provide rewards. The total number of states is therefore nxn + 1.
  6

4.2 Dynamic Programming
Dynamic programming involves breaking a problem into sub-problems and then recursively finding the solutions to the sub-problems. The problem at hand can be thought of as a Markov Decision Process where the Agent knows everything about its environment. Hence, dynamic programming can be used to find the solution. This involves the use of value functions to organize and structure the search for good policies.
Given that the action, and reward sets, S, A(s), and reward(R) for s ∈ S, are finite, and that its dynamics are given by a set of probabilities p(s’, r|s, a), for all s ∈ S, a ∈ A(s), r ∈ R, and s’ ∈ S+ (S+ is S plus a terminal state if the problem is episodic), Optimal policies can be easily obtained by finding the optimal value functions, V* or Q*, which satisfy the Bellman optimality equations:
V∗(s)=maxa􏰀a′,rp(s′,r|s,a)[r+γV∗(s)]........................... Eq4.1 Q∗(s,a)=maxa􏰀a′,rp(s′,r|s,a)[r+γmaxa′Q∗(s,a′)]........................... Eq4.2 Value of a state, V*: The optimal recursive sum of rewards starting from a state.
Q-value of a state, action pair, Q*(s,a): The optimal sum of discounted rewards associated
with a state-action pair.
4.3 Tabular model-based reinforcement learning
Tabular implies that the state-action space of the problem is small enough to fit in an array or a table and model-based reinforcement learning involves using experience to construct an internal model of the transitions and immediate outcomes in the environment. Appropriate actions are then chosen by searching or planning in this world model.
4.3.1 Policy evaluation:
Policy evaluation aims to compute the state-value function Vπ for an arbitrary policy . To produce each successive approximation, Vk+1 from Vk, iterative policy evaluation applies the same operation to each state S: it replaces the old value of S with a new value obtained from the old values of the successor states of S (in-place evaluation), and the expected immediate rewards, along all the one- step transitions possible under the policy being evaluated. It therefore returns a sequence V0, V1, . . . of estimates of Vπ.
Listing 1: Iterative policy evaluation
Input: π, the policy to be evaluated; theta, the tolerance limit; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
Initialize V(s) = 0, for all states in S Iterations = 0
Repeat
delta = 0
iterations += 1
for each state in S:
V(s) 􏰀π(s,a)􏰀′Pa′[Ra′+γV(s)] a s ss ss
delta = max (delta, |v V(s)|)
until delta < theta or iterations > max_iterations
Output: V = Vπ
  7

4.3.2 Policy Improvement:
Policy improvement attempts to find approximate solutions to the Bellman optimality equations (Eq 4.1, Eq 4.2). A policy π may be improved to a policy π′ with the equation below:
π(s)=argmaxaQπ(s,a)=argmaxa􏰀sPa′[Ra ′ +γVπ(s)]........................... Eq4.3 ss ss
Thereby picking the best action.
4.3.3 Policy Iteration:
This is completed using policy improvement and policy evaluation to produce a combined sequence ofpoliciesandvaluesintheformπ0,Vπ0,π1,Vπ1,π2,Vπ2,...πn,Vπn. Thismethodincrementally looks at the current values and extracts a policy. The last change to the actions will happen before the small rolling-average updates end. This is completed using policy improvement and policy evaluation.
4.3.4 Value Iteration
Value iteration learns the value of the states from the Bellman Update directly. The Bellman Update is guaranteed to converge to optimal values, under some non-restrictive conditions. Pseudocode in listing 2:
Listing 2: Value iteration pseudocode
Input: one-step dynamics (P and R); theta, the tolerance limit; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
Initialize V(s) = 0, for all states in S Iterations = 0
Repeat
delta = 0
iterations += 1
for each state in S:
V(s) 􏰀 π(s,a) 􏰀′ Pa′[Ra ′ +γV(s)] a s ss ss
delta = max (delta, |v V(s)|)
until delta < theta or iterations > max_iterations
for each state in S:
π(s)=argmaxa 􏰀′Pa′[Ra′+γV(s)]
 s ss ss
Output: optimal deterministic policy π when θ → 0.
4.4 Tabular model-free reinforcement learning
Model-free reinforcement learning uses experience to learn directly one or both of two simpler quantities (state/ action values or policies) which can achieve the same optimal behaviour without estimation or use of a world model. Given a policy, a state has a value, defined in terms of the future utility that is expected to accrue starting from that state.
 8

4.4.1 Sarsa Control
Intheabsenceoftheone-stepdynamics(Pass′ andRass′),aSarsa(State–Action–Reward–State–Action) agent interacts with the environment and updates the policy based on actions taken (on-policy). The Q value (Eq 4.2) for a state-action is updated by an error, adjusted by the learning rate (α).
Q(st,at)←Q(st,at)+α[rt+1+γQ(st+1,at+1)−Q(st,at)]........................... Eq4.4
The learning rate determines to what extent the newly acquired information overrides old infor- mation. A factor of 0 will make the agent not learn anything, while a factor of 1 would make the agent consider only the most recent information. Figure 3 demonstrates the sarsa control algorithm.
Figure 3: Sarsa Control Flowchart
 9

4.4.2 Q-learning control
Q-learning is a model-free algorithm that finds an optimal policy in the sense of maximizing the expected value of the total reward over any and all successive steps, starting from the current state. Q-learning can identify an optimal action-selection policy for any given FMDP, given infinite exploration time and a partly-random policy. Just like Sarsa, Q-Learning improves the estimate of the value of a state based on estimates of the values of other states (bootstrap) based on temporal differences (the difference between the immediate reward plus the estimated expected return from the next state and the estimated expected return for the current state)
Q(st,at)←Q(st,at)+α[rt+1+γmaxQ(st+1,a)−Q(st,at)]........................... Eq4.5
Figure 4: Q-learning Control Flowchart
 10

4.5 Non-tabular model-free reinforcement learning
Model-free methods can be combined with function approximation, making it possible to apply the algorithms to larger problems, even when the state space is continuous. One of such methods is linear regression. This involves making a prediction Y given an input X.
4.5.1 Sarsa control using linear function approximation:
Sarsa control can be adapted to use stochastic gradient descent to approximate action-value func- tions (Qπ).
4.5.2 Q-Learning control using linear function approximation:
One solution is to use an (adapted) artificial neural network as a function approximator. Function approximation may speed up learning in finite problems, due to the fact that the algorithm can generalize earlier experiences to previously unseen states.
5 Techniques Implemented
While most parts of the solution were implemented as suggested, slide modifications were made. The implementations are structued as follows.
5.1 Frozen Lake Environment
The grid of states were represented with the 4x4 matrix [[’&’, ’.’, ’.’, ’.’ ],
[’.’, ’#’, ’.’, ’#’ ],
[’.’, ’.’, ’.’, ’#’ ],
[’#’, ’.’, ’.’, ’$’ ]]
And the actions were depicted by a set of coordinates [(-1, 0), (0, -1), (1, 0), (0, 1)] translating to the indices of the transition directions.
&: Start state, no slip with Pass’ = 1 and Rass’ = 0
“.”: Frozen lake, Pass′ = 1 – slip + slip/4 for available actions and slip/4 for others, Rass′ = 0
#: Hole, transitions into the absorbing state with Pass′ = 1 and Rass′ = 0
$: Goal transitions into the absorbing state with Pass′ = 1 and Rass′ = 1
The actions were displayed as [’↑’, ’←’, ’↓’, ’→’] while rendering.
5.2 Tabular Model-Based Model
5.2.1 Policy Evaluation
Policy evaluation was implemented with the pseudocode in Listing 1.
5.2.2 Policy Improvement
Listing 3 is the pseudocode that improves the policy:
Listing 3: Policy Improvement
Input: V, the value function; γ, discount factor to ensure convergence 11
 
for each state in S:
π(s)=argmaxa 􏰀′Pa′[Ra′+γV(s)]
Output: π
5.2.3 Policy Iteration
Listing 4 shows the implementation of policy iteration
Listing 4: Policy Iteration
s ss ss
   5.2.4
Input: π, the policy to be evaluated; theta, the tolerance limit; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
Initialize V(s) = 0, for all states in S if no policy:
Initialize policy = 0 for all states
Iterations = 0
Repeat
delta = 0
iterations += 1
v_pi = policy_evaluation policy = policy_improvement delta = max (delta, |v_pi v|) v = v_pi
until delta < theta or iterations > max_iterations Output: policy, Vπ
Value Iteration
Value Iteration was implemented as shown in Listing 2
5.3 Tabular Model-Free Model
5.3.1 Sarsa Control
Sarsa control was implemented with the pseudocode shown in Listing 5 Listing 5: Sarsa Control
Input: eta, initial learning rate; epsilon, initial exploration factor; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
      Initialize eta with a random decision linspace(eta, 0, max_episodes)
      Initialize epsilon with a random decision linspace(epsilon, 0,
         max_episodes)
 12

 5.3.2
Initialize Q(s,a) ← 0 for all (s,a)
for each i in {1max_episodes}
s initial state for episode i
probability of taking a random action, p0 ← epsilon[i] Probability of using the argmax action of Q, p1 ← 1 - epsilon[i] d ← random choice (a=[0, 1], p=[p0, p1])
if d is 0:
select random action a
else:
a ← argmax Q
Repeat
Perform step and retrieve features and r, and also check if state
is terminal
d ← random choice (a=[0, 1], p=[p0, p1]) if d is 0:
         select random new action a’
      else:
a ← argmax Q
Q(s,a) += eta[i] x (r + γ x Q(s’,a’) Q(s,a))
s ← s’
a ← a’
Until terminal state
π ← argmax Q(:,a) V ← max Q(:,a)
Output: π, V
Sarsa Compare
In addition to the sarsa control function above, sarsa_compare receives optimal-value and max_iterations used to compare sarsa with optimal policy in the max_episodes loop.
Listing 6: Sarsa Compare
π = policy from sarsa
obtain Vπ by evaluating π with the policy evaluation function
delta = 0
for all states in S:
delta = max (delta, |v V(s)|) if delta < theta:
break the looping of max\_episodes
  13

5.3.3 Q-learning Control
The Q-learning algorithm was implemented as shown in Listing 7 Listing 7: Q-learning Control
  5.3.4
Input: eta, initial learning rate; epsilon, initial exploration factor; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
Initialize eta with a random decision linspace(eta, 0, max_episodes)
Initialize epsilon with a random decision linspace(epsilon, 0,
   max_episodes)
Initialize Q(s,a) ← 0 for all (s,a)
for each i in {1max_episodes}
s initial state for episode i
probability of taking a random action, p0 ← epsilon[i] Probability of using the argmax action of Q, p1 ← 1 - epsilon[i] d ← random choice (a=[0, 1], p=[p0, p1])
if d is 0:
select random action a
else:
a ← argmax Q
Repeat
Perform step and retrieve features and r, and also check if state
is terminal
d ← random choice (a=[0, 1], p=[p0, p1]) if d is 0:
         select random new action a’
      else:
a ← argmax Q
Q(s,a) += eta[i] x (r + γ x maxaQ(s’,:) Q(s,a))
s ← s’
a ← a’
Until terminal state
π ← argmax Q(:,a) V ← max Q(:,a)
Output: π, V
Q-learning Compare
Like with the sarsa_compare function, q_learning_compare receives optimal value and max_iterations used to compare sarsa with optimal policy in the max_episodes loop.
Listing 8: Q-learning Compare 14
 
π = policy from Q-learning
obtain Vπ by evaluating π with the policy evaluation function
delta = 0
for all states in S:
delta = max (delta, |v V(s)|) if delta < theta:
break the looping of max\_episodes
5.4 Non-Tabular Model-Free Model
The linear wrapper class have the methods encode_policy and decode_policy. Decode policy re- ceives a parameter vector “theta” obtained by a non-tabular reinforcement learning algorithm and returns the corresponding greedy policy, while The method encode state is responsible for repre- senting a state with the feature matrix. The methods reset and step return a feature matrix when they would typically return a state s. Each row a of this feature matrix contains the feature vector φ(s, a) that represents the pair of action and state (s, a).
5.4.1 Linear Sarsa
The following method was added to the linear wrapper
Listing 9: Linear Sarsa pseudocode
Input: eta, initial learning rate; epsilon, initial exploration factor; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
      Initialize eta with a random decision linspace(eta, 0, max_episodes)
      Initialize epsilon with a random decision linspace(epsilon, 0,
         max_episodes)
theta ← 0 for every feature for each i in {1max_episodes}
features initial state for episode i Q ← features dot theta
probability of taking a random action, p0 ← epsilon[i] Probability of using the argmax action of Q, p1 ← 1 - epsilon[i] d ← random choice (a=[0, 1], p=[p0, p1])
if d is 0:
select random action a
else:
a ← argmax Q
Repeat
Perform step and retrieve features and r, and also check if state
is terminal
Q ← features dot theta
d ← random choice (a=[0, 1], p=[p0, p1]) if d is 0:
               select random new action a
  15

 5.4.2
else:
a ← argmax Q
theta += eta[i] x (r + γ x Q(a) Q(a)) x features(a)
features ← features’ Q ← Q’
a ← a’
   Until terminal state
Output: theta
Linear Q-learning
The method in listing 10 was added to the linear wrapper
Listing 10: Linear Q-learning pseudocode
Input: eta, initial learning rate; epsilon, initial exploration factor; max_iterations, the limit to the number iterations; γ, discount factor to ensure convergence
      Initialize eta with a random decision linspace(eta, 0, max_episodes)
      Initialize epsilon with a random decision linspace(epsilon, 0,
         max_episodes)
theta ← 0 for every feature for each i in {1max_episodes}
features initial state for episode i Q ← features dot theta
probability of taking a random action, p0 ← epsilon[i] Probability of using the argmax action of Q, p1 ← 1 - epsilon[i] d ← random choice (a=[0, 1], p=[p0, p1])
if d is 0:
select random action a
else:
a ← argmax Q
Repeat
Perform step and retrieve features and r, and also check if state
is terminal
Q ←sdottheta
d ← random choice (a=[0, 1], p=[p0, p1]) if d is 0:
select random new action a else:
a ← argmax Q
            delta = r - Q[a]
             Q  = features dot theta
delta += γ x max(Q)
 16

            theta += eta[i] x delta x features[a]
features ← features
Q←Q
Until terminal state
      Output: theta
5.5 Main Function
This was implemented for execution as suggested, albeit with the inclusion of a big frozen lake grid.
6 Experimental Study
The following experiments are conducted:
1. Running tabular model based algorithm for the big lake. (Corresponding to Question 2)
2. Running Sarsa and Q-learning control for the small lake with comparison with the value of optimal policy. (Corresponding to Question 3)
3. Running Sarsa and Q-learning control for the big lake to find an optimal policy. (Correspond- ing to Question 5)
6.1 Tabular model for the big lake (Corresponding to Question 2)
In order to check which algorithm is faster, we added the checking of time-consuming: Listing 11: Time consuming check
         # Check time consuming of policy_iteration
start_time = time.time()
policy, value = policy_iteration(env, gamma, theta, max_iterations) print("--- {} seconds ---".format(time.time() - start_time))
         # Check time consuming of value_iteration
start_time = time.time()
policy, value = value_iteration(env, gamma, theta, max_iterations) print("--- {} seconds ---".format(time.time() - start_time))
The result are shown in table 1:
From table 1, we can see both policy iteration and value iteration got the optimal policy. The
optimal policies and the values of two algorithms are identical.
Although value iteration takes 20 loops to get the result, while it’s only 6 loops for policy
iteration, the running time of value iteration is less. The reason is that the policy evaluation and policy improvement are called in interleaved manner in policy iteration, and there is a loop in policy evaluation, but there’s no inner loop in value iteration. So value iteration runs faster. The following log shows the iteration times in the experiment:
   17

Table 1: Policy Iteration and Value Iteration Results
6.2 Sarsa and Q-learning for the small lake (Corresponding to Question 3)
6.2.1 Sarsa control with comparison
To check if the resulting policy of Sarsa is the optimal policy, we need to implement another function called sarsa_with_comparison in which an optimal policy is passed and the comparison of the value of each iteration and the optimal policy is conducted. The code in listing 12 is added at the end of each episode:
Listing 12: Optimality Check
# Compare with the value of optimal policy
# Get the current policy
policy = q.argmax(axis=1)
# Calculate the value of current policy by using policy_evaluation value = policy_evaluation(env, policy, gamma, theta_for_policy_eval,
            max_iterations_for_policy_eval)
         # Get the max difference of each state
delta = 0
for s in range(env.n_states - 1):
delta = max(delta, np.abs(value[s] - optimal_value[s])) # Check the delta with given threshold theta
if delta < theta_for_policy_eval:
found_optimal = True
print("Sarsa finds optimal policy in ", i + 1, " iterations") break
The calling of this function is shown in Listing 13:
Listing 13: Call of the sarsa_with_comparison function
      Policy Iteration
  Value Iteration
    Optimal Policy
         Value of policy
         Loop Times
 6
  20
    Running Time
 0.769460916519165 seconds
  0.5043950080871582 seconds
         18

         policy, value = sarsa_with_comparison(env, max_episodes, eta, gamma,
            epsilon, optimal_value, theta, max_iterations, seed=seed)
         env.render(policy, value)
The optimal_value is got by using value_iteration, and it returns the optimal policy and the value of it is as in Figure 5.
(a) Policy (b) Values of policy
Figure 5: Results of the Sarsa algorithm
The algorithm was run 10 times with the parameters: max_episode = 10000, eta = 0.5, epsilon = 0.5. The results gotten from each of the runs are displayed in table 2.
Table 2: Sarsa Experiment Results
        Run
 Policy
Value of Policy
 Result
   Number of Iterations
    1
    Valid Policy (Optimal)
   2305
    2
    Valid Policy (Optimal)
   2828
    3
    Valid Policy (Optimal)
   7336
    4
    Valid Policy (Optimal)
   5892
      19

     5
    Valid Policy (Not Optimal)
   10000
    6
    Valid Policy (Optimal)
   2083
    7
    Valid Policy (Not Optimal)
   10000
    8
    Valid Policy (Not Optimal)
   10000
    9
    Invalid Policy
   10000
    10
    Invalid Policy
   10000
      From table 2 we can see three different results: Valid policy(optimal), Valid policy(Not optimal) and Invalid policy.
• An invalid policy is a policy that can not guide the agent to get to the goal. There are 2 cases of getting an invalid policy in 10000 loops.
• In some cases, a valid policy can be got but it is not the same as the optimal policy, there are 3 cases in the experiment.
• There’re 5 times in which an optimal policy is got, the episodes numbers are listed in the table.
The rate of getting an optimal policy is 50% and the average episodes is 4088.8 in our experiment
6.2.2 Q-learning control with comparison
Same as Sarsa, we implemented q_learning_with_comparison function in which the same code is implemented. With the same parameters as 6.2.1, max_episode = 10000, eta = 0.5, epsilon = 0.5, 10 experiments are conducted, the results of which are shown in table 3. The rate of getting an optimal policy is 90% and the average episodes is 2219.8. Through this experiment we can see that Q-learning can get a better result than Sarsa.
20

Table 3: Q-learning Experiment Results
          Run
Policy
Value of Policy
Result
Number of Iterations
           1
Valid Policy (Optimal)
1243
           2
Valid Policy (Optimal)
2022
           3
Valid Policy (Optimal)
1258
           4
Valid Policy (Optimal)
3355
           5
Invalid Policy
10000
           6
Valid Policy (Optimal)
2017
           7
Valid Policy (Optimal)
4825
           8
Valid Policy (Optimal)
1134
           9
Valid Policy (Optimal)
2073
          10
Valid Policy (Optimal)
2051
            21
 
6.3 Sarsa and Q-learning for the big lake(Corresponding to Question 5)
6.3.1 Sarsa control for the big lake
In this experiment, we used the same way to check if an optimal policy can be got by using Sarsa control. With tweaking the parameters. The experiment result is listed in table 4.
Table 4: Sarsa with big lake Experiment Results
     Run
Max Episodes
  Learning Rate
 Exploration Factor
Run Time (seconds)
  Result
  Number of Iterations
    1
1000
  0.5
 0.5
33.17
  Invalid Policy
  1000
    2
1000
  0.7
 0.5
29.40
  Invalid Policy
  1000
    3
1000
  0.3
 0.5
28.84
  Invalid Policy
  1000
    4
1000
  0.5
 0.3
32.04
  Invalid Policy
  1000
    5
1000
  0.5
 0.7
28.60
  Invalid Policy
  1000
    6
3000
  0.5
 0.5
91.90
  Invalid Policy
  3000
    7
3000
  0.7
 0.5
88.42
  Invalid Policy
  3000
    8
3000
  0.3
 0.5
89.99
  Invalid Policy
  3000
    9
3000
  0.5
 0.3
91.30
  Invalid Policy
  3000
    10
3000
  0.5
 0.7
87.26
  Invalid Policy
  3000
    11
5000
  0.5
 0.5
148.92
  Invalid Policy
  5000
    12
5000
  0.7
 0.5
148.47
  Invalid Policy
  5000
    13
5000
  0.3
 0.5
147.45
  Invalid Policy
  5000
    14
5000
  0.5
 0.3
149.78
  Invalid Policy
  5000
    15
5000
  0.5
 0.7
147.89
  Invalid Policy
  5000
                 The optimal policy is not got after tuning the parameters. The running time becomes much longer when max_episode is adjusted bigger.
6.3.2 Q-learning control for the big lake
For this experiment we used the same methodology used in the one for Sarsa with the Big Lake environment, however, we tried to run the algorithm two more times with a much bigger Max Episode quantity. The results of this experiment are shown in Table 5.
According to the test result of 6.2, it should be easier to get an optimal policy when using Q-learning control. But it can be seen in the table above, an optimal was not got either even the max_episode was tuned to 30000. It took a lot of time to end the calculation. The reason why an optimal policy is so hard to be got in a big lake is that it’s hard to reach the goal position by random. All the rewards of all states will be 0 if the goal position is never reached in each episode, so an optimal policy can not be got easily. We tried to tune the max_steps of environment to be a bigger number, e.g. it was tuned to 500, to raise the probability of reaching the goal position, but the result is not got either. The result of each episode was printed, it shows that all the episodes were terminated because of dropping into a hole or the agent didn’t reach the goal position in 500 steps.
22

Table 5: Q-learning with big lake Experiment Results
     Run
Max Episodes
  Learning Rate
 Exploration Factor
Run Time (seconds)
  Result
  Number of Iterations
    1
1000
  0.5
 0.5
29.42
  Invalid Policy
  1000
    2
1000
  0.7
 0.5
30.53
  Invalid Policy
  1000
    3
1000
  0.3
 0.5
29.40
  Invalid Policy
  1000
    4
1000
  0.5
 0.3
30.47
  Invalid Policy
  1000
    5
1000
  0.5
 0.7
28.68
  Invalid Policy
  1000
    6
3000
  0.5
 0.5
94.70
  Invalid Policy
  3000
    7
3000
  0.7
 0.5
89.88
  Invalid Policy
  3000
    8
3000
  0.3
 0.5
103.42
  Invalid Policy
  3000
    9
3000
  0.5
 0.3
92.77
  Invalid Policy
  3000
    10
3000
  0.5
 0.7
91.30
  Invalid Policy
  3000
    11
5000
  0.5
 0.5
154.74
  Invalid Policy
  5000
    12
5000
  0.7
 0.5
154.94
  Invalid Policy
  5000
    13
5000
  0.3
 0.5
145.61
  Invalid Policy
  5000
    14
5000
  0.5
 0.3
156.14
  Invalid Policy
  5000
    15
5000
  0.5
 0.7
141.64
  Invalid Policy
  5000
    16
20000
  0.5
 0.5
596.14
  Invalid Policy
  20000
    16
30000
  0.5
 0.5
952.84
  Invalid Policy
  30000
                   23

7 Discussion
During the development of the project, we faced some problems which we managed to solve and others which either we knew the solution to but were unable to implement it or we weren’t able to resolve them at all.
1. After implementing the Sarsa and Q-learning, the results were always invalid policy with all of the actions being 0.
This problem is caused by inappropriate parameters. There are 2 parameters which caused the problem. The 1st parameter is max_steps in environment. It was set to be 16 at the beginning, but if this parameter is too small, a successful episode will be hard to be achieved. In most cases, the episode will be terminated due to reach the max_steps before reaching the goal position. So this parameter is tuned to 100 for the small lake. The 2nd parameter is max_episodes. Referring to the chapter 6.2, it is hard to get a valid policy if the max_episodes is small. This parameter was changed to 10000, then there was more chance to get a valid policy.
2. An optimal policy was not got when doing the Sarsa and Q-learning with com- parison with optimal policy.
An optimal policy was not got when calling sarsa_with_comparison. One parameter of this function is an optimal policy which will be compared with as the reference policy. In our case, the optimal policy was got by value_iteration. At last the reason of this problem was found. It was because the parameters gamma in value_iteration and sarsa_with_comparison were different (One was 0.9, the other one was 0.5). The function policy_evaluation was used in sarsa_with_comparison to get the value of all states. The value won’t be the same if gamma are different even if the input policies are the same.
3. How can each element of the parameter θ can be interpreted in linear action-value function approximation? (Corresponding to first part of Question 4)
In linear action-value function, the feature vector φ(s,a) is an array with the element of pair (s, a) is 1 and all the other elements are 0. The feature is a matrix with n_actions rows and n_actions * n_states columns. In Figure 6, the 1st term at the left hand is the feature
Figure 6: Matrix multiplication to get the Q-Values
 24

8
4.
matrix, for a given state s, e.g. state = 0, the value of F(0, 0), F(1, 1), F(2, 2) and F(3, 3) will be 1, all the other elements are 0. The right hand is the Q value of each action in state s. The parameter θ is a one dimension array with n_actions * n_states elements. So the value of θ is the Q value of each action in each state.
Why the tabular model-free algorithms are a special case of non-tabular model- free algorithm?(Corresponding to first part of Question 5)
The reason that the tabular model-free are a special case of the non-tabular model-free algo- rithms, is due to the way that the algorithm consider the states. In non-tabular model-free algorithms, states are defined as an infinite space represented by a feature vector φ(s)∈Rm, whereas in tabular model-free models, states have are finite, and thus, they can be represented by a simple array of dimensions number of states. This is the most important difference, as due to this, the value that is returned by the models changes, as it is meant for different purposes. In tabular models, the value that is returned is the array of Q-values for every state and action, because we know every state that there is in the environment, this is enough to specify a policy that would maximise the result upon reaching the goal. However, in the non-tabular models, since the number of states are theoretically infinite, it needs to account for any previously unseen states, thus, the value that is returned by the models is a set of values called θ, which can be used by the Value function to get the value of any state, V(s;θ). Thus, in essence, the main difference between the tabular models and non-tabular models, is in the amount of states that they have to consider when trying to find an optimal policy.
Conclusion and Future Work
In conclusion, when comparing model-based and model-free algorithms, we observed that given a specified environment, of which we know every state and have access to it, we know all the actions that can be taken in the environment and there is a goal for the agent to achieve, model-based algorithms work much better than model-free ones. This is due to the fact that a model-based has access to everything regarding the environment, and it can thus find the most optimal policy to reaching that goal just by using what it knows about the environment. Furthermore, model-free algorithms require a lot of interactions with the environment before being able to find the optimal policy. Thus, making them very time inefficient. However, the advantage of model-free algorithms, is that they can work even without knowing everything about the environment. This, becomes especially useful for non-tabular algorithms, as they won’t even know all the states possible in the environment. However, the main problem with model-free models, is the time taken for them to compute the optimal policy, since the bigger the environment the more time they will require to find the optimal policy.
Future work that could be done in the subject, could involve exploring how to reduce the computational time for model-free algorithms. It could also be possible to explore in the direction of mixing both model-based and model-free algorithms to find a middle ground of efficiency and adaptability.
References
[1] Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. “Reinforcement learning: A survey”. In: Journal of artificial intelligence research 4 (1996), pp. 237–285.
25

[2] Leemon C Baird III and Andrew W Moore. “Gradient descent for general reinforcement learn- ing”. In: Advances in neural information processing systems. 1999, pp. 968–974.
[3] Michail G Lagoudakis and Ronald Parr. “Least-squares policy iteration”. In: Journal of machine learning research 4.Dec (2003), pp. 1107–1149.
[4] Shane Legg. “Machine super intelligence”. PhD thesis. Università della Svizzera italiana, 2008.
[5] Jim X Chen. “The evolution of computing: AlphaGo”. In: Computing in Science & Engineering 18.4 (2016), pp. 4–7.
[6] Yang Gao et al. “Reinforcement learning from imperfect demonstrations”. In: arXiv preprint arXiv:1802.05313 (2018).
[7] David Silver et al. “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play”. In: Science 362.6419 (2018), pp. 1140–1144.
26


###
#
# Sussman Anomaly- Q Learning
#  This trial examins the performance of Q Learning on a problem demonstrating the
#  Sussman anomaly, to establish that neither QL nor Murin are subject to it.
#
###

#Standard
import random,math,time

#For array management
import numpy as np

###
# Machine Learning segment
###

#Class implementing standard Q Learning
class QLearning:

    def __init__(self,S,A,l,i=0):
        #Initialize state/action space

        self.sp = 0 #Previous states and actions
        self.ap = 0

        self.l = l #Learning rate
        self.S = S #State and action lists
        self.A = A

        #Initialization mode- 1 is random, otherwise weighted proportional
        if i == 0:
            self.Q = np.random.random((S,A))
        else:
            self.Q = (1.0/A)*np.ones((S,A))

    #Method to implement selecting an action
    def act(self,s):
        As = np.cumsum(self.Q[s,:]) #Cumsum over the state slice of Q array
        r = As[-1]*random.random() #random selection of number in cumsum
        sel = np.sum(1.0*(As < r)) #selection by sum of elements less than selected cumsum level
        self.sp = s #previous state set to state from which action taken
        self.ap = sel #previous action set to selected action
        return sel #Return the seleted action

    #Method to implement a training step
    def train(self,r,learn_law=None):
        #learn_law lets you put in a function of (Qp,l,r) that's not the basic learning rule
        
        Qp = self.Q[self.sp,self.ap] #Grab the current Q value

        if (learn_law != None): #For default learning scheme
            Qn = Qp + self.l*(r - Qp) #Update Q with standard rule
            self.Q[self.sp,self.ap] = round(Qn,3) #put into value- rounded for numeric stability

        else:
            Qn = learn_law(Qp,self.l,r) #apply alternate RL rule
            self.Q[self.sp,self.ap] = round(Qn,3) #put into value- rounded for numeric stability

#Function to generate a state from the block stacks
def state(table):
    S = 0
    if len(table[0])>0:
        S = S + table[0][0]
    if len(table[1])>0:
        S = S + 4*table[1][0]
    if len(table[2])>0:
        S = S + 16*table[2][0]
    return S

#Function to move a block
def move(m):
    table[m[1]] = table[m[1]] + (table[m[0]][-m[2]:])
    del table[m[0]][-m[2]:]

#Metric to implement reinforcement
def metric(table):
    score = 0
    if [A] in table:
        score+=1
    if [B] in table:
        score+=1
    if [C] in table:
        score+=1
    if [C,A] in table:
        score-=1
    if [B,A] in table:
        score+=2
    if [C,B] in table:
        score+=2
    if [C,B,A] in table:
        score+=100
    return score

#Sussman Experiment

#Block label states
A = 1
B = 2
C = 3

#Initial state of the table
table = [[B],[A,C],[]]

#Set of possible moves
moves = [(0,1,1),(0,2,1),(1,0,1),(1,2,1),(2,0,1),(2,1,1),
         (0,1,2),(0,2,2),(1,0,2),(1,2,2),(2,0,2),(2,1,2),
         (0,1,3),(0,2,3),(1,0,3),(1,2,3),(2,0,3),(2,1,3)]

#Create QL agent
agent = QLearning(58,18,0.3,10)

ct = 0 #step counter
t_avg = -1 #Rollaing average of ticks

#For up to 1001 ticks
while (ct < 1001):
    t = 0 #Set ticks to 0

    states = [] #State list
    while not([C,B,A] in table): #While not sorted
        sp = state(table) #Get state
        a = agent.act(sp) #Get the agent's action
        table_p = table #Note prior table
        move(moves[int(a)]) #Execute the moves
        s = state(table) #Get new state
        if (s not in states): #Train when discovering a new state- reward exploration
            agent.train(1)
            states.append(s) # Append discovered state
        else:
            agent.train(0) #Otherwise, neutral feedback
        t+=1

    #Increment counter and tick average
    ct+=1
    t_avg = (0.9*t_avg + 0.1*t)*(t_avg != -1) + (t)*(t_avg == -1)

    #Print count and tick average
    if ct%1 == 0:
        print(ct, t_avg)

    #Apply 30 random moves for next test configuration
    for i in range(30):
        r = random.randint(0,17)
        move(moves[r])


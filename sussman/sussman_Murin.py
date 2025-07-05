###
#
# Sussman Anomaly- Murin
#  This trial examins the performance of Murin on a problem demonstrating the
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


#Murin-based core Q-Learning class (adapted for outside support of variables)
class QLearningM:
    
    def __init__(self,S,A,l,i=0):
        #Initialize state number, action number, learning rate, and init mode

        self.l = l #Learning rate

        self.S = S #State and action numbers
        self.A = A

        #Initialize Q array- if 0, random, otherwise action-weighted
        if i == 0:
            self.Q = np.random.random((S,A))
        else:
            self.Q = (1.0/A)*np.ones((S,A))

    #Method to take an action
    def act(self,s,mode=1):
        if mode: #probabilistic
            As = np.cumsum(self.Q[int(s),:]) #Cumulative sum of Q array on action slice
            r = As[-1]*random.random() #Random index
            sel = np.sum(1.0*(As < r)) #Get action
        else: #maximum likelihoof
            m = np.max(self.Q[s,:]) #get max
            sel = max([i*(self.Q[s,i]==m) for i in range(np.shape(self.Q[s:s+1,:])[1])]) #Select max likelihood action
        return int(sel) #return selection


    #Method to train core QL network
    def train(self,r,sp,ap,d):
        Qp = self.Q[int(sp),int(ap)] #Grab prior Q value
        Qn = Qp*(1-self.l*d) + self.l*d*(r) #Update Q value- Murin assumes standard learning
        self.Q[int(sp),int(ap)] = Qn*(Qn >= 0) #update Q matrix, threshold negative values

#Class implementing the full Murin algorithm on top of the minimal QL class
# this version implements State/Action augmentation, observed to be most efficient
class Murin:

    #Initialize learner
    def __init__(self,S,A,l,m):

        #State and action numbers
        self.S = S
        self.A = A
        
        #Array of state/action pairs- **not including augmentation**
        #   M = [[s1,s2...]
        #        [a1,a2...]]

        self.M = np.zeros((2,m))
        self.m = m #Memory length

        #Build modified Q array in subclass
        self.Q = QLearningM(S*A,A,l,1) #SxA array holds concatenated classes

    #Method to take an action
    def act(self,s,mode=1):
        ap = self.M[1,0] #Grab previous action from memory
        sA = s + ap*self.S #Calculate augmeted state as stride-indexed number
        aS = self.Q.act(sA,mode=mode) #Pull action from subclass
        self.M[:,1:] = self.M[:,0:self.m-1] #Update memory with new values
        self.M[0,0] = sA #Update memory array
        self.M[1,0] = aS
        return aS #Return the action

    #Method for training
    def train(self,r):
        #Loops over the memory depth, updating each prior learning pair
        # with a decay discounted rate based on how far back it occurred
        for a in range(self.m):
            self.Q.train(r,self.M[0,a],self.M[1,a],1.0/(a+1))

#Murin implementation of stat/state linking (usually less efficient than action linking)
class QMS:

    def __init__(self,I,S,A,m,l):

        #State space variables
        self.I = I #state/state index length (non-augmented states)
        self.S = S
        self.A = A

        self.m = m #Memory depth
        self.l = l #Learning rate

        #Augmented state arrays
        self.aS = I*S #state/state
        self.aA = S*A #state/action (for updates)

        #Q array and memory list
        self.Q = np.ones((self.aS,self.aA))
        self.M = np.ones((2,m))*-1

        self.s = 0 #Initial 0-state for init- highlights the bootstrap problem for S/S linking

    #Method to take an action
    def act(self,i):
        si = i + self.I*self.s #augmented state from previous state as stride-indexed value
        As = np.cumsum(self.Q[si,:]) #get cumulative sum over state slice
        r = As[-1]*random.random() #random selection
        sel = np.sum(1.0*(As < r)) #select out the action index

        self.M[:,1:] = self.M[:,:-1] #update the memory array
        self.M[0,0] = si
        self.M[1,0] = sel

        self.s = state = i #update the state
        return sel #return the selected action

    #Method to do a training step
    def train(self,r):
        d = 1.0 #Initial decay parameter
        for a in range(self.m): #across the histore
            sp = self.M[0,n] #grab state and action
            ap = self.M[1,n]
            Qp = self.Q[sp,ap] #original Q value
            Qn = Qp*(1-self.l*d) + self.l*d*(r) #Update Q value
            self.Q[sp,ap] = Qn*(Qn >= 0) #Threshold and update
            d = d*0.5 #exponentially decreasing decay

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

#Create Murin Agent
agent = Murin(58,18,0.9,4)

#Sussman Experiment

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
    

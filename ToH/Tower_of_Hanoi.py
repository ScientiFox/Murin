###
#
# Tower of Hanoi - Murin
#  Example of Murin algorithm learning the Tower of Hanoi problem
#  This example learns on 5 rings
#
###


'''
Big problem with the part where 0 is
the bottom but 1-3 are the stacks either 3 needs
to be top and ring must be > dest
but using 0 as empty so quality scale is kinda
fucked. 

you will not go to space today.
'''

#Standard
import random,math,time

#For array management
import numpy as np

###
# Machine learning segment
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
        self.Q = QLearningM(S*A,A,l,0) #SxA array holds concatenated classes

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

#Variables for the puzzle
rings = 5 #Number of rings
towers = {1:[a for a in range(rings+1)],2:[0],3:[0]} #Initial configuration 
moves = {0:(1,2),1:(1,3),2:(2,1),3:(2,3),4:(3,1),5:(3,2)} #Possible moves
twrs = [a for a in range(rings+1)] #Towers representation

#A selection of different state-construction functions for experiments
def get_state_v1():
    s = [-1,-1,-1]
    s[0] = 1.0*(len(towers[1])>len(towers[2]))
    s[1] = 1.0*(len(towers[2])>len(towers[3]))
    s[2] = 1.0*(len(towers[3])>len(towers[1]))
    return sum([s[i]*(2**i) for i in range(len(s))])

def get_state_v2():
    s = [-1,-1,-1]
    s[0] = 1.0*(len(towers[1])>len(towers[2])) + 2.0*(len(towers[1])==len(towers[2]))
    s[1] = 1.0*(len(towers[2])>len(towers[3])) + 2.0*(len(towers[2])==len(towers[3]))
    s[2] = 1.0*(len(towers[3])>len(towers[1])) + 2.0*(len(towers[3])==len(towers[1]))
    return sum([s[i]*(3**i) for i in range(len(s))])

def get_state_v3():
    q = (towers[1][-1]) + (towers[2][-1])*(rings+1) + (towers[3][-1])*((rings+1)**2)
    return q


#A selection of different quality metrics for deriving rewards
def quality_v1():
    return sum(towers[1])*1 + sum(towers[2])*3 + sum(towers[3])*6

def quality_v2():
    return 3*sum(towers[3])-sum(towers[2])-sum(towers[1])

def quality_v3():
    d2 = sum([towers[1][a]-twrs[a] for a in range(len(towers[1]))])
    d2 = d2 + sum([towers[2][a]-twrs[a] for a in range(len(towers[2]))])
    d2 = d2 + sum([towers[3][a]-twrs[a] for a in range(len(towers[3]))])
    return d2

#Function to move a ring from one tower to another    
def move(frm,to):
    ring = towers[frm][-1] #Get top ring on from tower
    dest = towers[to][-1] #Get top ring on destination tower
    if ring > dest and ring != 0: #If viable mover
        del towers[frm][-1] #Remove top from from tower
        towers[to].append(ring) #Add to the to tower
        return 1 #Return a success
    else: #If not a valid move, return failure
        return 0    

#Function to display a tower state
def disp():
    print(towers[1])
    print(towers[2])
    print(towers[3])

#Tower of Hanoi Experiment:

#Make Murin agent
agent = Murin((rings+1)**3,6,0.2,5)

#Initialize towers
towers = {1:[a for a in range(rings+1)],2:[0],3:[0]}
hist = [] #History of steps to goal count
q10 = quality_v3() #Baseline quality metrics
q = quality_v3() 
qratio = 1.1 #Baseling quality ratio

hh = -1 #Rolling average history of counts
ct = 0 #Trial counter

#While the history average is over a threshold, is the first step, or less than 100 trials
while ((hh > 20.0) or (hh == -1) or (ct < 100)):

    hist = 0 #Set counter to 0
    prev = [] #Empty list of prior states
    towers = {1:[a for a in range(rings+1)],2:[0],3:[0]} #Reset towers puzle

    #While not a solution state
    while (towers[3] != [a for a in range(rings+1)]):

        #Sense/act Stage
        s = get_state_v2() #Make the state
        r = agent.act(s) #Get the agent action
        m = move(moves[r][0],moves[r][1]) #Make the chosen move

        #Training Stage
        q10 = q #Previous quality measure
        q = quality_v3() #New quality measure

        #Exploration reward training machanism
        if (s not in prev):
            agent.train(1)
            prev.append(s)
        else:
            agent.train(0)

        #Quality improvement training mechanism
        #if q > q10:
        #    agent.train(1)

        #Increment steps to goal counter
        hist+=1

    #If the first sample, set hh to counts value
    if hh == -1:
        hh = hist
    else: #Otherwise set to rolling average
        hh = 0.9*hh + 0.1*hist

    #Increment trial coutner
    ct +=1
    print(ct,hh) #Print results

#Print final count and history average
print(ct,hh)


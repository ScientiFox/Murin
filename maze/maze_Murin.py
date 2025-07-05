###
#
# Training the Murin algorithm to do localized maze-navigation
#
###

#Standard
import random,math,time

#For array management
import numpy as np

#For image outputs
import cv2

###
# Machine Learning Functions
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
    def act(self,s):
        As = np.cumsum(self.Q[int(s),:]) #Cumulative sum of Q array on action slice
        r = As[-1]*random.random() #Random index
        sel = np.sum(1.0*(As < r)) #Get action
        return sel #return it

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
    def act(self,s):
        ap = self.M[1,0] #Grab previous action from memory
        sA = s + ap*self.S #Calculate augmeted state as stride-indexed number
        aS = self.Q.act(sA) #Pull action from subclass
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

#A sample maze for exploration training
maze = np.array(
[[1,1,1,1,1,1,1,1,1,1,1,1,1,1],
 [1,0,1,0,0,1,0,1,0,0,0,0,0,1],
 [1,0,1,0,1,1,0,1,0,1,0,1,0,1],
 [1,0,1,0,0,0,0,1,1,1,0,1,0,1],
 [1,0,1,0,1,1,1,1,0,1,1,1,0,1],
 [1,0,0,0,0,0,0,1,0,0,0,1,0,1],
 [1,1,0,1,1,1,0,1,0,1,1,1,0,1],
 [1,1,0,1,0,0,0,1,0,0,1,0,0,1],
 [1,0,0,1,0,1,0,1,1,0,1,0,1,1],
 [1,1,0,1,0,1,1,1,1,0,1,0,1,1],
 [1,1,0,1,1,1,0,0,1,0,0,0,1,1],
 [1,0,0,0,0,1,0,1,1,0,1,0,0,1],
 [1,0,1,1,0,0,0,0,0,0,1,0,1,1],
 [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
 ])

#Orientation, postionm and 'sensor' directions
orientation = 0 #0-3: ^ > v <
pos = [1,1]
scan = [
    (1,0),
    (0,1),
    (1,2),
    (2,1),
    (1,0),
    (0,1)
    ]

#State-building function for the world model
def get_state(pos,ori):
    local = maze[int(pos[0])-1:int(pos[0])+2,int(pos[1])-1:int(pos[1])+2]
    scans = [(2**a)*local[scan[int(a+ori)][0],scan[int(a+ori)][1]] for a in range(3)]
    return sum(scans),[scans[i]>0 for i in range(len(scans))]

#Function to move the agent in the world
def move(a):
    a = (a-1)%4 #Grab direction coords
    dx = abs(a-2)-1 #Make change in x and y
    dy = 1 - abs(a-1)
    if maze[int(pos[0] + dy),int(pos[1] + dx)] != 1: #If wouldn't hit a wall
        pos[0] = pos[0] + dy #Change position
        pos[1] = pos[1] + dx
        return 1 #Note success
    else: #Otherwise return failure to move
        return 0

#An ideal comparison agent, designed to use the left hand rule
def ideal(sc,sp):
    L = sc[0] #Grab iterative parameters
    C = sc[1]
    R = sc[2]
    #Pick movement based on local state and last moves
    if (sp == 1):
        return 1,0
    elif (L == 0):
        return 0,1
    elif (C == 0):
        return 1,0
    elif (R == 0):
        return 2,0
    else:
        return 0,0

#Build the agent
M = 2 #memory depth
wrong_threshold = 0.0 #threshold change for a bad move
agent = Murin(8,3,0.99,M) #make the agent

#Data variables
ct_avg = -1 #Average counts to destination
epoch = 0 #epoch number

while (epoch < 1000): #Looping over 1k epochs

    #Set fresh position and orientation
    pos = [1,1]
    orientation = 2

    #Make a destination
    destination = [-1,-1]
    while maze[destination[0],destination[1]] == 1:
        destination = [random.randint(0,13),random.randint(0,13)]

    ct = 0 #counts reinitialized
    epoch+=1 #increment epoch number

    sp = 0 #previous state holder
    visited = [[pos,orientation,1]] #List of visited cells
    s,scans = get_state(pos,orientation) #Get new state data

    while pos != destination: #While navigating
        ct+=1 #Increment counter

        #sense/act segment
        s,scans = get_state(pos,orientation)
        a = agent.act(s)

        #Update positon and orientation from action
        pos_p = pos+[]
        if (a == 0)|(a == 2):
            orientation = (orientation+a-1)%4
            ri = 1
        else:
            ri = move(orientation)

        #train to search new spaces
        if [pos,orientation,1] in visited:
            if [pos,orientation,2] in visited:
                #agent.train(0) #Optional neutral reinforcement
                pass #pass otherwise
            else:
                visited.append([pos,orientation,2]) #Add to visited cells list
                agent.train(1) #Apply training
        else:
            visited.append([pos,orientation,1])
            agent.train(1)

    #Update count metric and print
    ct_avg = (ct_avg == -1)*(ct) + (ct_avg != -1)*(0.9*ct_avg + 0.1*ct)
    print(epoch,ct_avg)

###
#
# Experimenting with teaching the Murin agents to navigate on a grid
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


#Grid movement for different headings- h:(dx,dy)
move = {1:(0,1),2:(1,0),3:(0,-1),4:(-1,0)} #Movements
card_head = {1:"N",2:"E",3:"S",4:"W"} #Cardinal directions
Left = {1:4,2:1,3:2,4:3} #Left
Right = {1:2,2:3,3:4,4:1} #Right
Turnabout = {1:3,2:4,3:1,4:2} #Turnabout
moves = {'l':Left,'r':Right,'t':Turnabout}
acts = {0:'l',1:'r',2:'t'}

#A grid object to represent the world
class grid:

    def __init__(self):
        #Initialization for the world
        self.pos = (0,-2)
        self.di = 1

    def turn(self,t):
        #method to make a turn on the grid
        self.pos = (self.pos[0] + move[self.di][0],self.pos[1] + move[self.di][1])
        self.di = moves[t][self.di]

    def get_track(self):
        #grab the wold state
        return self.pos,self.di

    def set_pos(self,n_pos,n_di):
        #Set the world position
        self.pos = n_pos
        self.di = n_di

    def get_rel_loc(self,goal):
        #Calculate the relative location in the world to the goal
        dsgx = 1.0*(goal[0]-self.pos[0])
        dsgy = 1.0*(goal[1]-self.pos[1])
        dsgl = math.sqrt(dsgx**2 + dsgy**2)
        dp = dsgy/dsgl
        theta = math.acos(dp)
        return round((theta*(dsgx>=0) + (dsgx<0)*(2*math.pi - theta))/(2*math.pi/8))

    def get_dist_goal(self,goal):
        #Get the literal distance to the goal
        dsgx = 1.0*(goal[0]-self.pos[0])
        dsgy = 1.0*(goal[1]-self.pos[1])
        return math.sqrt(dsgx**2 + dsgy**2)


#Main loop
if __name__ == '__main__':
    #states 9x for relative location to target
    #4x for orientation wrt location.
    #agent = Murin(9*4,3,0.95,2) <- reference

    #Make the agent
    agent = Murin(9*4,3,0.95,2) 

    world = grid() #make the world
    goal = ((1,1),1) #set the goal location
    dist = world.get_dist_goal(goal[0]) #Get initial measure information
    dh = dist #set tracking distance
    ori = world.get_rel_loc(goal[0]) #Get orientation to goal
    dref = dist #Initial reference distance
    ticks = 0 #Counting ticks running
    epochs = 0 #epoch counter
    ticklist = [] #List of ticks to solve

    #Loop over 500 epochs
    while epochs < 500:
        ticks+=1 #Increase tick each step

        #Act- grab state and get action
        s = world.get_rel_loc(goal[0]) + 9*(world.get_track()[1]-1)
        A = agent.act(s)
        world.turn(acts[A]) #Apply act to world

        #Check if goal
        if world.get_dist_goal(goal[0]) == 0:
            ticklist.append(ticks) #Add ticks to list
            if epochs > 0: #If latter runs
                try: #Grab the performance metric
                    measure = measure*0.9 + 0.1*ticklist[-1]
                except: #Otherwise, add first tick list
                    measure = ticklist[-1]
                print(epochs,round(measure,2)) #Print performance so far

            ticks = 0 #Reset ticks
            epochs+=1 #increment epochs
            world.set_pos((0,0),1) #reset world
            goal = world.get_track() #get new goal

            while goal[0] == world.get_track()[0]: #Don't let the next goal be the current start!
                goal = ((random.randint(-2,2),random.randint(-2,2)),random.randint(1,4))

            #New reference distance
            dref = 1.0*world.get_dist_goal(goal[0])
            dcnt = 0

        #Otherwise, if too far away
        elif world.get_dist_goal(goal[0]) >= 100:
            print(epochs,measure,"fell off the world") #Note it
            ticks = 0 #reset zero
            epochs+=1 #increment epochs
            world.set_pos((0,-2),1) #reset world
            goal = world.get_track() #get new track
            while goal[0] == world.get_track()[0]: #Make sure it's a new destination
                goal = ((random.randint(-2,2),random.randint(-2,2)),random.randint(1,4))
            dref = 1.0*world.get_dist_goal(goal[0]) #set new reference distance
            dcnt = 0

        #Training if not at end
        dp = world.get_dist_goal(goal[0]) #Get distance and orientation
        dr = 2*world.get_track()[1]
        if (dp < dist): #If improving, positive reinforcement
            r = 1
        elif (dp > dist+0.3): #Negative if more than a trivial loss
           r = -1
        else:
            r = 0 #if trival loss, neutral

        dist = dp #Get new distance and orientation
        ori = dr
        agent.train(r) #Apply training

    
    

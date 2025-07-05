###
#
# Murin - Sequential Q-Learning
#  This software implements the STQL/Murin algorithm, a machine learning
#  algorithm designed specifically for learning chained, temporal, behavior-
#  based action sequences for problem solving. It is designed to emulate
#  the training procedure associated with Operant Conditioning.
#
#  Building on a core basis of standard Q-Learning, it incorporates a
#  temporal learning element in the form of a concatenated state vector.
#  this augmentation allows it to learn more sophisticated patterns
#  of response to state conditions. Additionally, a decaying, historical
#  weight is applied to the learning rate extending back through the
#  recent history of the learning sequence, to enforce temporal patterning
#  It is therefor able to learn non-fixed transfer functions faster than
#  standard Q-Learning, as well as solve problems which Q Learning is unable
#  to adequately solve.
#
#  It can operate by concatenating previous state or previous actions to the
#  augmented state vector.
#
#  The Murin algorithm can effectively be deployed as an agent-based problem
#  solving system, with an instance activated until the conditions of a solution
#  are met. The brain cna be saved for later use on similar problems if entry
#  conditions are monitored for it, creating a database of solutions keyed to
#  problems, or else as a general enter-solve-abandon ad hoc framework.
#
#  This package implements standard Q-learning and the Murin algorithm.
#
###

#Standard
import random,math,time

#For array management
import numpy as np

#For image outputs
import cv2

#Standard sigmoid for IO normalization
def sigmoid_01(x,lmbda):
    return (1.0 / (1 + math.e**(-1.0*lmbda*x)))

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


# A set of three XOR functions, for testing very basic learning and validation
def XOR(a,b):
    return 0.0*(a != b) + 1.0*(a == b)

def XOR2(a,b):
    X = 0.0*(a != b) + 1.0*(a == b)
    return X + 2*(1-X)

def XOR3(a,b):
    X = 0.0*(a != b) + 1.0*(a == b)
    return X + 2*(1-X) + 4*(X)

if __name__ == '__main__':

    #lists of learning rates for testing
    #cL = [ (10.0**(a+1) - 1)/10**(a+1) for a in range(5)]
    cL = [0.95,0.9,0.8,0.7,0.5]

    #Array of reinforcement results for display
    A = np.zeros((101,301,3))

    #Image plotting
    for j in range(14):
        A[:,20*(j+1),2] = 254
    for j in range(80):
        y = 0.25*(j)
        A[100-int(5*y),j,1] = 254

    #For each learning rate to test
    for c in cL:

        #List of reinforcements for data collection
        r_list = [0]*300

        #Looping over 20 trials
        for k in range(20):
            #Indices and values for XOR test input
            n = 0
            a = 0
            b = 0
            ct = 0 #count

            #Make an agent learner
            #M = QMS(4,2,8,2,c) #QMS Learner
            #M = QLearning(4,4,c,1) #Standard QL learner
            M = Murin(10,8,c,1)

            Q = [0]*20 #Rolling average of results queue

            while((n < 300)):

                #Test over random or sequence inputs
                #a,b = random.randint(0,1),random.randint(0,1)
                a = 1 - a
                b = a*(1 - b) + (1 - a)*(b)
                o = M.act(a+2*b) #Simple state is the binary of a and b

                #calculate the value of the test function output and generate reward
                if o == XOR3(a,b):
                    r = 1
                else:
                    r = -1

                #update rolling average queue
                Q[:-1] = Q[1:]
                Q[-1] = r*(r>0)

                M.train(r) #train the agent

                r_list[n] = r_list[n] +sum(Q)*0.05 #update reward list, scaled to 20 step window

                n = n + 1 #Increment counter

        #Print data up to now
        print(c,r_list[-1])
        for a in range(len(r_list)):
            A[100-int(5*r_list[a]),a,0] = 255 #Update trial graph

        #Display 
        cv2.imshow("window",A)
        cv2.waitKey(2000) #view data for 2s

cv2.destroyAllWindows() #Clear windows

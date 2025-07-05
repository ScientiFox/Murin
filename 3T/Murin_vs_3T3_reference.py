###
#
# Reference implementation of Murin learning to play Tic Tac Toe
#
###

#Standard
import random,math,time

#For array management
import numpy as np

###
# Tic Tac Toe functions
###

#Wrapper function to concatenate x and y on axis a
def cona(x,y,a):
    return np.concatenate((x,y),axis=a)

#Function to get the sums of player marks on all lines in the board
def check_lines(B):

    #Rows and Columns
    h = np.array([np.sum(B,axis=0)])
    v = np.array([np.sum(B,axis=1)])

    #diagonals
    d1 = B[0:1,0:1] + B[1:2,1:2] + B[2:3,2:3]
    d2 = B[0:1,2:3] + B[1:2,1:2] + B[2:3,0:1]

    #output numbers arranged
    out = cona(cona(d2,h,1),cona(d1,v,1),1)

    #Return output
    return out

#function to get the state of the board encoded
def get_state(board):
    B = np.reshape(board,(1,9))+1 #flatten
    out = [B[0,i]*3**i for i in range(9)] #encode
    return sum(out) #return

#Function to identify if a cell must be blocked to win
def block(board):
    B = np.copy(board) #Copy of the board
    p = 2*(np.sum(B==1)<=np.sum(B==-1))-1 #decode whose turn it is
    act_list = [] #holder for actions list

    j = 0 #index incrementor
    for a in np.reshape(1*(B==0)*act_sel,(1,9))[0,:]: #looping over the flattened board array
        if a != 0: #If not a 0 entry
            act_list.append(j) #add that cell to the viable actions
        j+=1 #increment counter

    #For each action
    for a in act_list:
        Bh = np.copy(B) #copy the board

        #Get stride coords
        cx = a%3 
        cy = int(a/3)

        #Set coord to player move
        Bh[cy,cx] = p
        Chp = check_lines(Bh) #check if winner from move
        num2s = np.sum([-2*p == j for j in Chp]) #calculate number of 2-columns created
        if num2s == 0: #if none, return cell
            return a

#Function to make a move 'a' on a board 'B'
def move(B,a):
    p = 2*(np.sum(B==1)<=np.sum(B==-1))-1 #Calculate player
    Bh = np.copy(B) #copy board

    #Get coords
    cx = a%3 
    cy = int(a/3)

    Bh[cy,cx] = p #set spot to player id
    return Bh #Return new board

#An optimal player move selector for the given board
def optimal3(board):

    B = np.copy(board) #New copy
    p = 2*(np.sum(B==1)<=np.sum(B==-1))-1 #calculate player
    Ci = check_lines(B) #Check lines across board

    #list of actions and moves to make
    act_list = []
    move_list = []

    j = 0 #cell incrementor
    for a in np.reshape(1*(B==0)*act_sel,(1,9))[0,:]: #For each cell in flattened board
        if a != 0: #If not moved into 
            act_list.append(j) #append to available actions
        else:
            move_list.append(j) #otherwise append to moves made
        j+=1 #increment counter

    #If only the center square is taken
    if move_list in [[4]]:
        r = int(4*random.random())  #pick a random move from corners
        mvs = [0,2,6,8]
        return mvs[r],p #return moves and turn taken
    elif move_list in [[],[0],[2],[6],[8],[1],[3],[5],[7]]: #if no moves or non-center only
        mvs = [4] #take the center
        return mvs[int(1*random.random())],p #return move

    #Wins check
    mustBlock = -1 #Flag for opponent win conditions available
    for a in act_list: #Looping over all available actions
        Bh = np.copy(B) #Copy board

        cx = a%3 #get coords
        cy = int(a/3)
        Bh[cy,cx] = p #set to current player

        #check lines for player move
        Chp = check_lines(Bh)

        #Check for opponent's move
        Bh[cy,cx] = -1*p
        Cho = check_lines(Bh)

        #If player can win, return move
        if (3*p in Chp):
            return a,p
        if (-3*p in Cho): #If opponent can win, must block
            mustBlock = a

    #if any block condition was found, take it
    if mustBlock != -1:
        #print "Block win catch",mustBlock
        return mustBlock,p

    #2s checks

    cands = [] #List of candidates
    max2s = 0 #Most 2s created by a move

    for a in act_list: #For every available action

        Bh = np.copy(B) #copy board
        cx = a%3 #get coords
        cy = int(a/3)

        #Mark position with player action and check lines
        Bh[cy,cx] = p
        Chp = check_lines(Bh)

        #Find the number of 2s created by this move
        num2s = np.sum([2*p == j for j in Chp])
        if num2s > max2s: #Update tracker if more available, and reset candidates list
            cands = [a]
            max2s = num2s
        elif num2s == max2s: #if same number, add to candidates list
            cands = cands + [a]

    if max2s == 2: #If you can make 2 rows of 2, any is a win, take the first one in list
        return cands[0],p
    elif max2s == 1: #Else, if you can only make lines of 1

        removes = [] #filtered set of candidates
        for a in cands: #Checking all candidates

            Bh = np.copy(B) #copy board

            cx = a%3 #Get coords
            cy = int(a/3)
            Bh[cy,cx] = p #set to player

            ao = block(Bh) #Calculate if it's a blocking maneuver

            cx = ao%3 #get coords of blocking cell
            cy = int(ao/3)
            Bh[cy,cx] = -1*p #set to opponent move
            Cho = check_lines(Bh) #check lines
            num2so = np.sum([-2*p == j for j in Cho]) #check how many 2-lengths blocked
            if num2so == 2: #if it makes 2 to block, add it to the ones to remove
                removes.append(a)

        for a in removes: #remove all moves that leave a blocking condition intact
            cands.remove(a)

        if cands != []: #If solutions, return a random one, they're equally good
            return cands[int(len(cands)*random.random())],p

    #If no way to make an unblockable move
    elif max2s == 0:
        for a in act_list: #for each action

            Bh = np.copy(B) #copy board

            cx = a%3 #get coords
            cy = int(a/3)
            Bh[cy,cx] = -1*p #mark coords as opponent

            Cho = check_lines(Bh) #check lines

            num2so = np.sum([-2*p == j for j in Cho]) #check how many 2s your opponent could make there
            if (num2so == 2):  #Pick  move eliminating opponent's options
                return a,p

    #If nothing found, return a totally random move- this should never happen.
    return act_list[int(len(act_list)*random.random())],p

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

###
# Multiple different state abstractions for experiments
###

def get_state_7(B):
    Br = np.reshape(B,(1,9))+1
    out = [(Br[0,i]==0)*2**i for i in range(9)]
    return sum(out)

def get_state_6(B):
    h = np.array([np.sum(B,axis=0)])
    v = np.array([np.sum(B,axis=1)])
    d1 = B[0:1,0:1] + B[1:2,1:2] + B[2:3,2:3]
    d2 = B[0:1,2:3] + B[1:2,1:2] + B[2:3,0:1]
    out = cona(cona(d1,h[0:1,1:2],1),cona(v[0:1,1:2],d2,1),1)+1
    out = [out[0,i]*3**i for i in range(4)]
    return sum(out)

def get_state_5(B):
    #h = np.array([np.sum(B,axis=0)])
    v = np.array([np.sum(B,axis=1)])
    d1 = B[0:1,0:1] + B[1:2,1:2] + B[2:3,2:3]
    out = cona(d1,v,1)+1
    out = [out[0,i]*3**i for i in range(4)]
    return sum(out)

def get_state_4(B):
    #h = np.array([np.sum(B,axis=0)])
    v = np.array([np.sum(B,axis=1)])
    d1 = B[0:1,0:1] + B[1:2,1:2] + B[2:3,2:3]
    d2 = B[0:1,2:3] + B[1:2,1:2] + B[2:3,0:1]
    out = cona(d1,cona(v,d2,1),1)+1
    out = [(int(2.0*((out[0,i]+3.0)/6.0)))*3**i for i in range(6)]
    return sum(out)

def get_state_3(B):
    h = np.array([np.sum(B,axis=0)])
    d2 = B[0:1,2:3] + B[1:2,1:2] + B[2:3,0:1]
    v = np.array([np.sum(B,axis=1)])
    out = cona(h,cona(d2,v,1),1)+1
    out = [(int(2.0*((out[0,i]+3.0)/6.0)))*3**i for i in range(6)]
    return sum(out)

def get_state_2(board):
    B = np.reshape(board,(1,9))+1
    out = [(B[0,i] != 1)*2**i for i in range(9)]
    return sum(out)

def get_state(board):
    B = np.reshape(board,(1,9))+1
    out = [B[0,i]*3**i for i in range(9)]
    return sum(out)

#Make lists of moves and actions
moves = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
act_sel = np.reshape(np.array([[a+1 for a in range(9)]]),(3,3))

depth = 5 #Memory depth for agent
cycle = 1 #Trainign cycle variable
epochs_max = 85600 #Maximum training epochs
menu = [0] #optional multi value testing

#Select options for the menu plans
for a in menu:
    if a == 0:
        agent = Murin(3**10,9,0.9,depth)
        print("Murin "+str(agent.Q.l))
        print("------------")
    if a == 1:
        agent = Murin(3**10,9,0.1,depth)
        print("Murin "+str(agent.Q.l))
        print("------------")
    if a == 2:
        agent = QLearning(3**10,9,0.3,depth)
        print("QLearning "+str(agent.l))
        print("------------")
    if a == 3:
        agent = QLearning(3**10,9,0.4,depth)
        print("QLearning "+str(agent.l))
        print("------------")
    if a == 4:
        agent = QLearning(3**10,9,0.6,depth)
        print("QLearning "+str(agent.l))
        print("------------")

    #Experiment trial values
    deviance = 0.0
    epochs = 0
    winner = [0,0,0]

    #Looping over max epochs
    while(epochs < epochs_max):

        B = np.zeros((3,3)) #Make board
        acts = [0,1,2,3,4,5,6,7,8] #action list
        k = 2*random.randint(0,1)-1 #Player index toggle
        won = 0 #Winner note
        p = 1 #Player flag
        move_list = [] #List of moves
        agent_moves = 0 #Moves from the agent

        #Looping over 9 actions each game
        for ap in range(9):

            if (p == k): #If on the first player turn
                agent_moves+=1 #mark up agent turn
                ####################
                s = get_state(B) #Here's where you add in experimental state value
                ####################
                agent.m = depth #Set agent memory depth
                a = agent.act(s) #Take agent action
                while not(a in acts): #While not a valid action
                    agent.m = 1 #set shallow depth for training
                    agent.train(0) #Tell it not to make illegal moves
                    agent.m = depth #reset depth
                    a = agent.act(s) #get new action

                act_p = a #Grab action
                B[moves[int(a)][0],moves[int(a)][1]] = p #set to actually picked action

                i = sum([b*(acts[b]==a) for b in range(len(acts))]) #Remove action
                del acts[i]

            #for other player side
            if (p == -1*k):
                a,pi = optimal3(B) #Get optimal move

                ri = random.random() #random number
                if (ri < deviance): #If less than the variation chance
                    a = acts[random.randint(0,len(acts)-1)] #make random action for learning value

                act_p = a #Grab action
                act_pp = act_p  #previous
                act_ppp = act_pp #previous previous

                B[moves[a][0],moves[a][1]] = p #Set move on board

                i = sum([b*(acts[b]==a) for b in range(len(acts))]) #remove action
                del acts[i]

            move_list.append(int(a)) #Add selected move for list analysis
            C = check_lines(B) #Check lines

            if np.max(C) == 3: #If agent is the winner
                won = 1 #Mark winner
                winner[1-k] = winner[1-k] + 1 #update list for player order
                if (1-k == 0): #if first move
                    agent.m = agent_moves #set depth to number of moves
                    for a in range(cycle): #For how many cycles chosen to test
                        agent.train(1) #Train
                    agent.m = depth #reset depth
                if (1-k == 2): #If second mover
                    agent.m = agent_moves #depth to number of moves
                    for a in range(cycle): #across the cycle
                        agent.train(0) #train
                    agent.m = depth #reset depth
                break #End on end of game
            if np.min(C) == -3: #if losing
                won = 1 #mark winner
                winner[1+k] = winner[1+k] + 1 #update turn order
                if (1+k == 0): #train for first mover
                    agent.m = agent_moves
                    for a in range(cycle):
                        agent.train(1)
                    agent.m = depth
                if (1+k == 2): #Train for second mover
                    agent.m = agent_moves
                    for a in range(cycle):
                        agent.train(0)
                    agent.m = depth
                break #break on end of game
            p = -1*p #toggle player order otherwise

        if won != 1: #Not on a win
            winner[1] = winner[1] + 1 #Mark up tie list

            agent.m = agent_moves #set memory depth
            lorig = agent.Q.l #save learn rate
            for a in range(cycle): #over cycles, train
                agent.train(1)
            agent.m = depth #reset depth

        epochs+=1 #Epoch increase
        if epochs%100 == 0: #Very 100 epochs, print current data
            print(epochs,winner[0],winner[1],winner[2],deviance)
            winner = [0,0,0] #reset winners list




###
#
# 3T Long test
#  A trial running over a long series to unit test the Tic Tac Toe agents
#
###

#Standard
import random,math,time

#For array management
import numpy as np

#For loading brains
import pickle

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

#Set up list of moves and actions
moves = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
act_sel = np.reshape(np.array([[a+1 for a in range(9)]]),(3,3))

#Number of tests to run
tests = 10000
i = 0 #index counter

#Experiment result variables
awins = 0
ties = 0
rwins = 0

#Timing measure variable
t0 = time.time()

#Looping over the number of tests
while (i < tests):

    B = np.zeros((3,3)) #Set up the board
    acts = [0,1,2,3,4,5,6,7,8] #action list
    k = 2*random.randint(0,1)-1 #player order flag
    movesG = [] #Moves made list
    p = 1 #Player flag

    Ch = check_lines(B) #Check the lines
    while (0 in B) and not(3 in Ch) and not(-3 in Ch): #While moves available and no winner
        if (p == k): #
            #Agent
            a,pd = optimal3(B) #optimal agen move
            cx = a%3 #Get coords
            cy = int(a/3)
            B[cy,cx] = p #Move on board
            acts.remove(a) #Take that choice out
            movesG.append(a) #Add to moves taken
        elif (p == -1*k): #Other player
            #Random move
            r = int(len(acts)*random.random())
            a = acts[r]
            cx = a%3 #Get coords
            cy = int(a/3)
            B[cy,cx] = p #Make move on board
            acts.remove(a) #Take that move out of options
            movesG.append(a) #Add to moves made
        p = -1*p #Toggle player
        Ch = check_lines(B) #Check lines
    if (3*k in Ch): #If agent winner, count up
        awins+=1
    elif (-3*k in Ch): #If non-agent winner, report move sequence for analysus
        print(movesG)
        rwins+=1 #Count up
        break #Stop execution
    elif not(0 in B): #If a tie, count up
        ties+=1
    i+=1 #test index increment
    if (i%100)==0: #Report every 100 tests
        Tr = (((time.time()-t0)/i)*(tests - 1.0*i))/(60.0) #time estimate
        print(i,str(int(Tr))+"'"+str(int(60*(Tr-int(Tr))))+"\"") #output data

#Print data output
print(awins,ties,rwins)
print("Random trials: ",tests)
print("%wins: ",awins*100.0/tests)
print("%ties: ",ties*100.0/tests)
print("%losses: ",rwins*100.0/tests)

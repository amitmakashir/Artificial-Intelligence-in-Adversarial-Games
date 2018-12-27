#!/usr/bin/env python3

import sys

'''

Heuristic/Evaluation Function:
There are (in general) 3 cases around which I have designed my Evaluation function:

1) Goal States: If you have won the game i.e a goal state is found for you, 
                assign a big positive value to it, thus encouraging this move.
                But if you find a goal state where your opponent wins, 
                assign a big negative value and thus discouraging this move.
                This big value could be (n^3)*100.
                
2) Blocked the opponent:    For a 3x3 board, if for any row/col/diagonal the opponent 
                            has 2 pieces on it and we have 1 piece, then this successor has just saved us 
                            the game. Assign a big value just smaller than Goal state value.                                             

3) Other cases: If the board is not in the above two scenario, then assign +1 to all your pieces 
                on the top nxn board and -1 to opponent pieces. Calculate the sum of a row and
                multiply this sum by n. If the row is dominated by opponent the sum could a negative value.
                Multiplying by n will help give values in larger range [-(n-1)^n ,(n-1)^n] instead of 
                [-(n-1) , n-1] and as n grows this difference grows.
                Repeat the summing for all rows, columns and diagonals and then add these to get 
                the final score.
                Also I am checking for column balance. That means I am penalizing moves that involve dropping
                lot of pebbles in just a few columns. I am encouraging moves that involve 
                spreading your pebbles across columns because I think that would maximize 
                my chances of victory.
                                           
          
Minimax implementation:
1)  To always have a solution ready, I am iteratively increasing the depth or horizon of my search till h=15. 

2)  I have implemented minimax with alpha beta pruning. I kept a "nodes" variable which counts the 
    leaf nodes of my search tree(the nodes/board where evaluation function was called) for every depth 
    and I noticed a 30-40% decrease in the explored nodes.
    
3)  One of my concerns was the iterative process of searching the same tree with increased depth 
    wasn't very efficient. I remember professor saying you'll need an oracle to tell you which nodes 
    to explore first or arranging the nodes of max and min in descending and ascending order respectively.
    I thought about implementing dynamic programming, but as the depth is increased every time, the leaf 
    nodes are different and hence the value that will be propagated through those nodes will be different.
    A solution to this problem could be, exploring the nodes according to their alpha beta values in 
    the previous iterations. This has a good probability of maximizing the pruning of the tree and reducing 
    time complexity to almost O(b^(h/2)). As h increases the value of b^h becomes a very large value and a small
    increase in pruning efficiency would give great results.
    

General implemenation:
A board is represented by a 2d list of its columns. 
For eg. [[first column],[second column],[third column]] for a 3x3 board   
It was easier to drop a pebble or rotate nth column this way.

'''


def human_readable(board):
    n = len(board)
    for i in range(n + 3):
        print([board[j][i] for j in range(n)])


# Convert board array back to string
def getBoardString(boardState,n):
    board = boardState[0]
    board_string = ""
    for i in range(n+3):
        for j in range(n):
            board_string += board[j][i]
    return board_string


# Make array out of board string
def getBoardArray(board_string,n):
    board = []
    for i in range(n):
        col = [board_string[j] for j in range(i, len(board_string), n)]
        board.append(col)
    return board


# Put a pebble in col passed as second argument
def fill_col(boardState, col, player):
    board = boardState[0]
    total_moves = boardState[1]
    col_len = len(board[0])

    for i in reversed(range(col_len)):
        if board[col][i] == ".":
            board[col][i] = player
            boardState = [board, total_moves + [col + 1]]
            return boardState


# Rotate the col passed as second argument
def rotate_col(boardState, col, player):
    board = boardState[0]
    total_moves = boardState[1]
    board_col = board[col]
    index = -1
    for i in range(len(board_col)):
        if board_col[i] == ".":
            index = i
    element = board_col.pop()
    board_col.insert(index + 1, element)
    board[col] = board_col

    boardState = [board, total_moves + [-(col + 1)]]
    return boardState


def count_piece(board, col):
    counter = 0
    for i in board[col]:
        if i != ".":
            counter += 1
    return counter



def count_your_pieces(board,player):
    counter = 0
    for i in board:
        for j in i:
            if j == player:
                counter += 1

    return counter


# Returns a 2-d array where the first element is the board and the second element is the move to get this board
# Input : [board,path]
# Output : [ [ [board_i], move_i ],....]
# Note: The col used in fill_col and rotate_col are 0 indexed numbers, add 1 (or -1)to it readable while printing
def successor(boardState, player):
    successors = []

    board = boardState[0]
    n = len(board)
    # Drop a pebble in every column that is empty (n moves) and rotates every column (n moves)
    for col in range(n):
        if (count_piece(board, col) < n + 3 and count_your_pieces(board,player) < 20):
            new_board = [i[:] for i in board]
            new_boardState = [new_board, boardState[1]]
            successors.append(fill_col(new_boardState, col, player))

    for col in range(n):
        if count_piece(board, col) > 0:
            new_board = [i[:] for i in board]
            new_boardState = [new_board, boardState[1]]
            successors.append(rotate_col(new_boardState, col, player))

    return successors



#===========================================================

def is_goal(boardState, current_player):
    board = boardState[0]
    n = len(board)
    # Check all columns for goal (Check only top n)
    for col in board:
        if (len(set(col[:n])) == 1 and set(col[:n]) == {current_player}):
            return True

    # Check all rows for goal
    for i in range(n):  # row number
        row = [board[j][i] for j in range(n)]
        if (len(set(row)) == 1 and set(row) == {current_player}):
            return True

    # Check for diagonals
    diagonal = [board[i][i] for i in range(n)]
    if (len(set(diagonal)) == 1 and set(diagonal) == {current_player}):
        return True

    diagonal = [board[-(i + 1)][i] for i in range(n)]
    if (len(set(diagonal)) == 1 and set(diagonal) == {current_player}):
        return True

    return False


# ==============================================================

# Logic
# Assign +1 to all pieces of current player and -1 to opponent pieces in top n by n board
# If sum(row(i)) == n , multiply it by n*n
# H(s) = sum(all rows) + sum(all columns) + sum(all diagonals)

def evaluate(boardState,current_player):
    global nodes
    nodes += 1

    # Change to value to something that changes linearly with the n of the input
    big_value = n * n * n * 100.0

    if current_player == player:
        if (is_goal(boardState, player)):
            return big_value + (1.0 / len(boardState[1]))

        if (is_goal(boardState, other_player)):
            return -(big_value + (1.0 / len(boardState[1])))

    if current_player == other_player:
        if (is_goal(boardState, other_player)):
            return -(big_value + (1.0 / len(boardState[1])))

        if (is_goal(boardState, player)):
            return big_value + (1.0 / len(boardState[1]))


    board = replacePiecesandShape(boardState[0], player)
    score = rowScore(board) + colScore(board) + diagScore(board) + spreadyourPieces(board)

    global opponentBlocked
    if (opponentBlocked):
        opponentBlocked = False
        return big_value - 1.0

    return score


def replacePiecesandShape(board, player):
    n = len(board)
    new_board = []
    for i in range(n):
        row = []
        for col in board:
            if col[i] == player:
                row.append(1)
            elif col[i] == ".":
                row.append(0)
            else:
                row.append(-1)
        new_board.append(row)
    return new_board


def rowScore(board):
    n = len(board)
    score = 0
    for i in board:
        is_opponentBlocked(i)
        rowSum = sum(i)
        score += rowSum * n
    return score


def colScore(board):
    n = len(board)
    score = 0
    for i in range(n):
        col = [board[j][i] for j in range(n)]
        is_opponentBlocked(col)
        colSum = sum(col)
        if(colSum < 0):
            colSum = colSum * (1.3*n)
        else:
            colSum = colSum * (0.7*n)
        score += colSum
    return score


def diagScore(board):
    n = len(board)

    diagonal_1 = [board[i][i] for i in range(n)]
    is_opponentBlocked(diagonal_1)
    score1 = sum(diagonal_1) * n

    diagonal_2 = [board[n - i - 1][i] for i in range(n)]
    is_opponentBlocked(diagonal_2)
    score2 = sum(diagonal_2) * n

    return score1 + score2


# Checks the count of column and returns a value depending on balance of the col
def colBalance(col, n):
    total_allowed_pieces = 0.5 * n * (n + 3)
    your_piece = [i for i in col if i == 1]
    balanced_count = total_allowed_pieces / n
    return (n * 10) / (abs(balanced_count - len(your_piece) + 1))


# Check for the condition where opponent has n-1 pieces and you have 1 piece
# So you just saved the game. Assign this a very high score (just lower than goal state)
# parameter = row/col/diagonal
def is_opponentBlocked(parameter):
    if ((0 not in parameter) and (sum(parameter) == -(n - 1))):
        global opponentBlocked
        opponentBlocked = True
        return True
    return False

def spreadyourPieces(board):
    n = len(board)
    total_allowed_pieces = 0.5 * n * (n + 3)
    balanced_count = total_allowed_pieces / n
    score = 0

    for col in board:
       pieces_in_col = sum([1 for i in col if i == player])
       score += (n * 100) / (abs(balanced_count - pieces_in_col + 1))

    return score



# ===========================================================
# http://giocc.com/concise-implementation-of-minimax-through-higher-order-functions.html
# https://www.youtube.com/watch?v=STjW3eH0Cik

def minimax(boardState, player):
    board = boardState[0]

    if is_goal(boardState, player):
        # print(player + " has won this game")
        return 0

    #     if is_goal(boardState,"o"):
    #         print("Min has won this game")
    #         return 0

    successors = successor(boardState, player)
    best_move = successors[0][1][-1]  # successors -> [[[board-i],move-i], ...]

    succ_boardStates = [i for i in successors]  # Make a list of just the boards

    alpha, beta = float('-inf'), float('inf')

    for curr_boardState in succ_boardStates:
        score = min_turn(curr_boardState, alpha, beta)

        if score > alpha:
            # Only pick the first element of the list where current board matches i.e get the move that got you here
            best_move = curr_boardState[1][-1]
            # print("best move changed to:" + str(best_move))
            alpha = score

    return best_move


def min_turn(boardState, alpha, beta):
    if len(boardState[1]) > h or is_goal(boardState, "o") or is_goal(boardState, "x"):
        # print("Max has won this game")
        return evaluate(boardState,player)

    successors = successor(boardState, other_player)

    succ_boardStates = [i for i in successors]  # Make a list of just the boards

    for curr_boardState in succ_boardStates:
        beta = min([beta, max_turn(curr_boardState, alpha, beta)])

        if alpha >= beta:
            return beta

    return beta


def max_turn(boardState, alpha, beta):
    if len(boardState[1]) > h or is_goal(boardState, "x") or is_goal(boardState, "o"):
        return evaluate(boardState,other_player)

    successors = successor(boardState, player)

    succ_boardStates = [i for i in successors]  # Make a list of just the boards

    for curr_boardState in succ_boardStates:
        alpha = max([alpha, min_turn(curr_boardState, alpha, beta)])

        if alpha >= beta:
            return alpha

    return alpha

# ===========================================================
# Actual program starts from here

# These are inputs to the program
n = int(sys.argv[1])
player = sys.argv[2]
board_string = sys.argv[3]
time_limit = int(sys.argv[4])

col_len = n + 3

# Generate these variables from the inputs
board = getBoardArray(board_string,n)

boardState = [board, []]

# Set the other player
if player == "o":
    other_player = "x"
else:
    other_player = "o"

# Initial depth to be explored
h = 1
opponentBlocked = False

# Cut off at h
while h <= 15:
    # nodes explored
    nodes = 0
    move = minimax(boardState, player)

    # Copy the current board into new one
    new_board = [i[:] for i in boardState[0]]
    newBoardState = [new_board,[]]

    # Print the best move and board string
    if move > 0 :
        print(str(move) + " " + getBoardString(fill_col(newBoardState, move-1, player), n))
    elif move < 0 :
        print(str(move) + " " + getBoardString(rotate_col(newBoardState, abs(move+1), player), n))
    elif move == 0:
        print("0 " + getBoardString(newBoardState,n))

    h += 1



'''
jlundbl1_KInARow.py
Authors: Lundblad, Julia

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
import game_types
from winTesterForK import winTesterForK
from game_types import State, Game_Type
GAME_TYPE = None

AUTHORS = 'Julia Lundblad' 
USE_LLM = True

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# set to false if you can't get gemini LLM imports to work.
if USE_LLM == True:
    # for LLM
    import os
    import google.generativeai as genai
    # pip install -q -U google-generativeai
    genai.configure(api_key="(insert api key here)")

    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    model = genai.GenerativeModel(
        model_name="gemini-pro",
        safety_settings=safety_settings,
        generation_config=generation_config,
        # system_instruction="Make sure all your responses include some interesting information about AI technology, no matter what character you are playing."
    )

    def prompt(query):
        input = query
        response= model.generate_content(input)
        return response.text

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Cookie'
        if twin: self.nickname = 'Harry'
        self.long_name = 'The Cookie Monster'
        if twin: self.long_name = 'Harry Potter'
        self.persona = 'funny'
        if twin: self.persona = 'carefree'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".

    def introduce(self):
        if USE_LLM == True:
            try:
                intro = prompt("introduce yourself as if you are the cookie monster, and are about to play someone in K-in-a-row (a version of tic tac toe). make this intro only two sentences.")
                if self.twin: intro = prompt("introduce yourself as if you are Harry Potter, and are about to play someone in K-in-a-row (a version of tic tac toe). make this intro only two sentences.")
            except:
                intro = "non-LLM intro: I am " + self.long_name + ". Try your best to defeat me."
        else:
            intro = "I am COOKIE MONSTER! You can never defeat me.."
            if self.twin: intro = "I am Harry Potter, the boy who lived! I am a wizard, so you will have to try extra hard to defeat me in this game."
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.
        utterances_matter=True):      # If False, just return 'OK' for each utterance.

        # Write code to save the relevant information in variables
        # local to this instance of the agent.
        # Game-type info can be in global variables.

        # new code:
        self.who_i_play = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.time_limit = expected_time_per_move
        global GAME_TYPE
        GAME_TYPE = game_type
        print("this is julia's new bot playing ", game_type.long_name)
        return "OK"
   
    # The core of your agent's ability should be implemented here:             
    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        print("makeMove has been called")
        best_move = [0, 0]
        best_score = 0

        # get the depth remaining at the state (number of possible moves left)
        depth_remaining = 0
        for i in range(GAME_TYPE.n):
            for j in range(GAME_TYPE.m):
                if currentState.board[i][j] == ' ':
                    depth_remaining += 1

        if GAME_TYPE == game_types.FIAR:
            if depth_remaining > 4:
                depth_remaining = 4

        if GAME_TYPE == game_types.Cassini:
            if depth_remaining > 4:
                depth_remaining = 4

        best_move, best_score = self.minimax(currentState, depth_remaining, True, -10000, 10000)

        newState = State(currentState)
        if currentState.whose_move == 'X':
            newState.board[best_move[0]][best_move[1]] = 'X'
            newState.whose_move = 'O'
        else:
            newState.board[best_move[0]][best_move[1]] = 'O'
            newState.whose_move = 'X'

        utterance = ""
        if USE_LLM == True:
            try:
                utterance = prompt("pretend you are " + self.long_name + " and respond to someone who just said this: " + currentRemark + ". make this only three sentences.")
            except:
                utterance = "utterance quota reached, retry."
        else:
            utterance = "This isn't a very interesting utterance without the LLM!"
        return [[best_move, newState], utterance]

    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            pruning=False,
            alpha=None,
            beta=None,
            lastMove=None,
            zHashing=None):

        # check if winstate
        if lastMove != None:
            tempState = State(state)
            if winTesterForK(tempState, lastMove, GAME_TYPE.k) != 'No win':
                return [-1, 1], self.staticEval(state)

        # check for no moves left
        if depthRemaining == 0:
            return [-1, -1], self.staticEval(state)
        
        # maximizing player
        if state.whose_move == 'X':
            best = -10000
            move = [-1, -1]
            for i in range(GAME_TYPE.n):
                if beta <= alpha:
                    break
                for j in range(GAME_TYPE.m):
                    if state.board[i][j] == ' ':
                        next_state = State(state)
                        next_state.board[i][j] = 'X'
                        next_state.whose_move = 'O'
                        next_score = self.minimax(next_state, depthRemaining - 1, True, alpha, beta, [i, j])[1]
                        if next_score > best:
                            move = [i, j]
                            best = next_score
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
                if beta <= alpha:
                    break
            return move, best
        # minimizing player
        else:
            best = 10000
            move = [-1, -1]
            for i in range(GAME_TYPE.n):
                if beta <= alpha:
                    break
                for j in range(GAME_TYPE.m):
                    if state.board[i][j] == ' ':
                        next_state = State(state)
                        next_state.board[i][j] = 'O'
                        next_state.whose_move = 'X'
                        next_score = self.minimax(next_state, depthRemaining - 1, True, alpha, beta, [i, j])[1]
                        if next_score < best:
                            move = [i, j]
                            best = next_score
                        beta = min(beta, best)
                        if beta <= alpha:
                            break
        return move, best

 
    def staticEval(self, state):
        # Values should be higher when the states are better for X,
        # lower when better for O.

        # static eval for tic tac toe
        board = state.board
        if GAME_TYPE == game_types.TTT:
            # no depth limit for this board, so we can just return a score when win

            x1 = x2 = x3 = o1 = o2 = o3 = 0

            for r in range(GAME_TYPE.n):
                numx = numo = 0
                for c in range(3):
                    if board[r][c] == 'X':
                        numx += 1
                    if board[r][c] == 'O':
                        numo += 1
                if numx == 1 and numo == 0:
                    x1 += 1
                elif numx == 2 and numo == 0:
                    x2 += 1
                elif numx == 3 and numo == 0:
                    x3 += 1
                elif numo == 1 and numx == 0:
                    o1 += 1
                elif numo == 2 and numx == 0:
                    o2 += 1
                elif numo == 3 and numx == 0:
                    o3 += 1

            for c in range(GAME_TYPE.m):
                numx = numo = 0
                for r in range(3):
                    if board[r][c] == 'X':
                        numx += 1
                    if board[r][c] == 'O':
                        numo += 1
                if numx == 1 and numo == 0:
                    x1 += 1
                elif numx == 2 and numo == 0:
                    x2 += 1
                elif numx == 3 and numo == 0:
                    x3 += 1
                elif numo == 1 and numx == 0:
                    o1 += 1
                elif numo == 2 and numx == 0:
                    o2 += 1
                elif numo == 3 and numx == 0:
                    o3 += 1
            
            # check diagonols
            numx = numo = 0
            if board[0][0] == 'X':
                numx += 1
            elif board[0][0] == 'O':
                numo += 1
            if board[1][1] == 'X':
                numx += 1
            elif board[1][1] == 'O':
                numo += 1
            if board[2][2] == 'X':
                numx += 1
            elif board[2][2] == 'O':
                numo += 1
            if numx == 1 and numo == 0:
                x1 += 1
            elif numx == 2 and numo == 0:
                x2 += 1
            elif numx == 3 and numo == 0:
                x3 += 1
            elif numo == 1 and numx == 0:
                o1 += 1
            elif numo == 2 and numx == 0:
                o2 += 1
            elif numo == 3 and numx == 0:
                o3 += 1

            numx = numo = 0
            if board[2][0] == 'X':
                numx += 1
            elif board[2][0] == 'O':
                numo += 1
            if board[1][1] == 'X':
                numx += 1
            elif board[1][1] == 'O':
                numo += 1
            if board[0][2] == 'X':
                numx += 1
            elif board[0][2] == 'O':
                numo += 1
            if numx == 1 and numo == 0:
                x1 += 1
            elif numx == 2 and numo == 0:
                x2 += 1
            elif numx == 3 and numo == 0:
                x3 += 1
            elif numo == 1 and numx == 0:
                o1 += 1
            elif numo == 2 and numx == 0:
                o2 += 1
            elif numo == 3 and numx == 0:
                o3 += 1
            
            ret = ((100 * x3) + (10 * x2) + x1 - (100 * o3) - (10 * o2) - o1)
            return ret
        
        elif GAME_TYPE == game_types.FIAR:
            # check all lines
            #for r in range(GAME_TYPE.n - ):
            x1 = x2 = x3 = x4 = x5 = o1 = o2 = o3 = o4 = o5 = 0

            # check boarder lines (rows of 5)
            for r in range(7):
                numx = numo = 0
                for i in range(1, 6):
                    if board[r][i] == 'X':
                        numx += 1
                    elif board[r][i] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            for r in range(1, 6):
                numx = numo = 0
                for i in range(5):
                    if board[r][i] == 'X':
                        numx += 1
                    elif board[r][i] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            for r in range(1, 6):
                numx = numo = 0
                for i in range(2, 7):
                    if board[r][i] == 'X':
                        numx += 1
                    elif board[r][i] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            for c in range (7):
                numx = numo = 0
                for i in range(1, 6):
                    if board[i][c] == 'X':
                        numx += 1
                    elif board[i][c] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            for c in range (5):
                numx = numo = 0
                for i in range(5):
                    if board[i][c] == 'X':
                        numx += 1
                    elif board[i][c] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            for c in range (5):
                numx = numo = 0
                for i in range(2, 7):
                    if board[i][c] == 'X':
                        numx += 1
                    elif board[i][c] == 'O':
                        numo += 1
                if numo == 0:
                    if numx == 1:
                        x1 += 1
                    elif numx == 2:
                        x2 += 1
                    elif numx == 3:
                        x3 += 1
                    elif numx == 4:
                        x4 += 1
                    elif numx == 5:
                        x5 += 1
                elif numx == 0:
                    if numo == 1:
                        o1 += 1
                    elif numo == 2:
                        o2 += 1
                    elif numo == 3:
                        o3 += 1
                    elif numo == 4:
                        o4 += 1
                    elif numo == 5:
                        o5 += 1

            # check diagonols
            numx = numo = 0
            for i in range(1, 6):
                if board[i][i] == 'X':
                    numx += 1
                elif board[i][i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][i - 1] == 'X':
                    numx += 1
                elif board[i][i - 1] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i + 1][i - 1] == 'X':
                    numx += 1
                elif board[i + 1][i - 1] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][i + 1] == 'X':
                    numx += 1
                elif board[i][i + 1] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i - 1][i + 1] == 'X':
                    numx += 1
                elif board[i - 1][i + 1] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][6 - i] == 'X':
                    numx += 1
                elif board[i][6 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][5 - i] == 'X':
                    numx += 1
                elif board[i][5 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i - 1][5 - i] == 'X':
                    numx += 1
                elif board[i - 1][5 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][6 - i] == 'X':
                    numx += 1
                elif board[i][6 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(1, 6):
                if board[i][7 - i] == 'X':
                    numx += 1
                elif board[i][7 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1
            
            numx = numo = 0
            for i in range(1, 6):
                if board[i + 1][7 - i] == 'X':
                    numx += 1
                elif board[i + 1][7 - i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1
        
            ret = ((100*x5) + (50*x4) + (30*x3) + (10*x2) + (x1) - (100*o5) - (50*o4) - (30*o3) - (10*o2) - (o1))
            return ret
        
        elif GAME_TYPE == game_types.Cassini:
            # check all horizonal rows of 5
            x1 = x2 = x3 = x4 = x5 = o1 = o2 = o3 = o4 = o5 = 0

            for shift in range(4):
                for r in range(2):
                    numx = numo = 0
                    for i in range(5):
                        if board[r][i + shift] == 'X':
                            numx += 1
                        elif board[r][i + shift] == 'O':
                            numo += 1
                    if numo == 0:
                        if numx == 1:
                            x1 += 1
                        elif numx == 2:
                            x2 += 1
                        elif numx == 3:
                            x3 += 1
                        elif numx == 4:
                            x4 += 1
                        elif numx == 5:
                            x5 += 1
                    elif numx == 0:
                        if numo == 1:
                            o1 += 1
                        elif numo == 2:
                            o2 += 1
                        elif numo == 3:
                            o3 += 1
                        elif numo == 4:
                            o4 += 1
                        elif numo == 5:
                            o5 += 1

            for shift in range(4):
                for r in range(2):
                    numx = numo = 0
                    for i in range(5):
                        if board[r + 5][i + shift] == 'X':
                            numx += 1
                        elif board[r + 5][i + shift] == 'O':
                            numo += 1
                    if numo == 0:
                        if numx == 1:
                            x1 += 1
                        elif numx == 2:
                            x2 += 1
                        elif numx == 3:
                            x3 += 1
                        elif numx == 4:
                            x4 += 1
                        elif numx == 5:
                            x5 += 1
                    elif numx == 0:
                        if numo == 1:
                            o1 += 1
                        elif numo == 2:
                            o2 += 1
                        elif numo == 3:
                            o3 += 1
                        elif numo == 4:
                            o4 += 1
                        elif numo == 5:
                            o5 += 1

            for shift in range(3):
                for c in range(2):
                    numx = numo = 0
                    for i in range(5):
                        if board[i + shift][c] == 'X':
                            numx += 1
                        elif board[i + shift][c] == 'O':
                            numo += 1
                    if numo == 0:
                        if numx == 1:
                            x1 += 1
                        elif numx == 2:
                            x2 += 1
                        elif numx == 3:
                            x3 += 1
                        elif numx == 4:
                            x4 += 1
                        elif numx == 5:
                            x5 += 1
                    elif numx == 0:
                        if numo == 1:
                            o1 += 1
                        elif numo == 2:
                            o2 += 1
                        elif numo == 3:
                            o3 += 1
                        elif numo == 4:
                            o4 += 1
                        elif numo == 5:
                            o5 += 1

            for shift in range(3):
                for c in range(2):
                    numx = numo = 0
                    for i in range(5):
                        if board[i + shift][c + 5] == 'X':
                            numx += 1
                        elif board[i + shift][c + 5] == 'O':
                            numo += 1
                    if numo == 0:
                        if numx == 1:
                            x1 += 1
                        elif numx == 2:
                            x2 += 1
                        elif numx == 3:
                            x3 += 1
                        elif numx == 4:
                            x4 += 1
                        elif numx == 5:
                            x5 += 1
                    elif numx == 0:
                        if numo == 1:
                            o1 += 1
                        elif numo == 2:
                            o2 += 1
                        elif numo == 3:
                            o3 += 1
                        elif numo == 4:
                            o4 += 1
                        elif numo == 5:
                            o5 += 1
            
            # check diagonols
            numx = numo = 0
            for i in range(5):
                if board[i][3 + i] == 'X':
                    numx += 1
                elif board[i][3 + i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(5):
                if board[2 + i][i] == 'X':
                    numx += 1
                elif board[2 + i][i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(5):
                if board[4 - i][i] == 'X':
                    numx += 1
                elif board[4 - i][i] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            numx = numo = 0
            for i in range(5):
                if board[6 - i][i + 3] == 'X':
                    numx += 1
                elif board[6 - i][i + 3] == 'O':
                    numo += 1
            if numo == 0:
                if numx == 1:
                    x1 += 1
                elif numx == 2:
                    x2 += 1
                elif numx == 3:
                    x3 += 1
                elif numx == 4:
                    x4 += 1
                elif numx == 5:
                    x5 += 1
            elif numx == 0:
                if numo == 1:
                    o1 += 1
                elif numo == 2:
                    o2 += 1
                elif numo == 3:
                    o3 += 1
                elif numo == 4:
                    o4 += 1
                elif numo == 5:
                    o5 += 1

            ret = ((100*x5) + (50*x4) + (30*x3) + (10*x2) + (x1) - (100*o5) - (50*o4) - (30*o3) - (10*o2) - (o1))
            return ret

        return 0

# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:26:15 2019

@author: Ahmad Mohammad
Professor: Dr. Phillips, Intro to Artificial Intelligence
Middle Tennessee State University
Illustration of a set

Portions based on Python code provided by
Scott P. Morton
Center for Computational Science
Middle Tennessee State University
"""
import sys
import copy
import heapq

class PriorityQueue():
    def __init__(self):
        self.thisQueue = []
    def push(self, thisNode):
        heapq.heappush(self.thisQueue, (thisNode.val, -thisNode.id, thisNode))
    def pop(self):
        return heapq.heappop(self.thisQueue)[2]
    def isEmpty(self):
        return len(self.thisQueue) == 0
    def length(self):
        return len(self.thisQueue)

nodeid = 0
class node():
    def __init__(self, val, table, parent=None):
        global nodeid
        self.id = nodeid
        nodeid += 1
        self.val = val
        self.table = table
        self.parent = parent
        self.g = nodeid  # Cost from start node to current node
        self.h = 0  # Heuristic estimate from current node to goal
        
    def __str__(self):
        return 'Node: id=%d val=%d'%(self.id, self.val)


class Set():
    def __init__(self):
        self.thisSet = set()
    def add(self,entry):
        if entry is not None:
            self.thisSet.add(entry.__hash__())
    def length(self):
        return len(self.thisSet)
    def isMember(self,query):
        return query.__hash__() in self.thisSet


class state():
    def __init__(self):
        self.xpos = 0
        self.ypos = 0
        self.tiles = [[0,1,2],[3,4,5],[6,7,8]]
    def left(self):
        if (self.ypos == 0):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos][s.ypos-1]
        s.ypos -= 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def right(self):
        if (self.ypos == 2):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos][s.ypos+1]
        s.ypos += 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def up(self):
        if (self.xpos == 0):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos-1][s.ypos]
        s.xpos -= 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def down(self):
        if (self.xpos == 2):
            return None
        s = self.copy()
        s.tiles[s.xpos][s.ypos] = s.tiles[s.xpos+1][s.ypos]
        s.xpos += 1
        s.tiles[s.xpos][s.ypos] = 0
        return s
    def __hash__(self):
        return (tuple(self.tiles[0]),tuple(self.tiles[1]),tuple(self.tiles[2]))
    def __str__(self):
        return '%d %d %d\n%d %d %d\n%d %d %d\n'%(
                self.tiles[0][0],self.tiles[0][1],self.tiles[0][2],
                self.tiles[1][0],self.tiles[1][1],self.tiles[1][2],
                self.tiles[2][0],self.tiles[2][1],self.tiles[2][2])
    def copy(self):
        s = copy.deepcopy(self)
        return s



def a_star_algo(initial_state, goal_state, heuristic_function):
    # Initialize data structures for the A* search
    open_list = PriorityQueue()
    closed_set = Set()
    

    
    # Create a node for the initial state
    initial_node = node(heuristic_function(initial_state, goal_state), initial_state)
    initial_node.g = 0
    initial_node.f = initial_node.val
    initial_node.table = initial_state
    

    

    #print(heuristic_function(initial_state, goal_state))
    # Push the initial node into the open list
    open_list.push(initial_node)
    
    # Initialize statistics
    nodes_visited = 0
    max_nodes_in_memory = 0
    depth = 0
    optimal_path = []
    current_state = initial_state
    while not open_list.isEmpty():
        nodes_visited += 1
        # Calculate the current number of nodes in memory
        current_nodes_in_memory = open_list.length() + closed_set.length()

        # Update max_nodes_in_memory if the current number is greater
        max_nodes_in_memory = max(max_nodes_in_memory, current_nodes_in_memory)

        # Get the node with the lowest f(n) from the open list
        current_node = open_list.pop()
        # When you expand a node, increase the depth by 1
        depth += 1

        # Check if the current state is the goal state
        if are_states_equal(current_node.table, goal_state):
            # Reconstruct the solution path
            path = []
            while current_node is not None:
                path.insert(0, current_node.table)
                current_node = current_node.parent
            optimal_path = path

            result = {
                'nodes_visited': nodes_visited,
                'max_nodes_in_memory': max_nodes_in_memory,
                'path': optimal_path,  # Add the solution path to the result
                # Add more statistics as needed
            }
            return result

        
        #Mark the current state as visited
        closed_set.add(current_node.table)
        # Generate successor states
        successor_states = [current_node.table.up(), current_node.table.down(),
                            current_node.table.left(), current_node.table.right()]

        
        for successor_state in successor_states:
            #print(successor_state) []
            if successor_state is not None and not closed_set.isMember(successor_state):
                # Create a node for the successor state
                successor_node = node((heuristic_function(successor_state, goal_state) + current_node.g), successor_state, current_node)                      

                open_list.push(successor_node)
        
    # If the search is exhausted without finding the goal state, return None
    return None

# Had to add this or got error
def are_states_equal(state1, state2):
    for i in range(3):
        for j in range(3):
            if state1.tiles[i][j] != state2.tiles[i][j]:
                return False
    return True

def column_distance_heuristic(state, goal):
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state.tiles[i][j]
            goal_position = [(r, c) for r in range(3) for c in range(3) if goal.tiles[r][c] == tile][0]
            distance += abs(j - goal_position[1])  # Calculate the column distance
    return distance


def manhattan_distance(state, goal):
    # Calculate the Manhattan distance for each tile from the goal state
    distance = 0
    for i in range(3):
        for j in range(3):
            tile = state.tiles[i][j]
            goal_position = [(r, c) for r in range(3) for c in range(3) if goal.tiles[r][c] == tile][0]
            distance += abs(i - goal_position[0]) + abs(j - goal_position[1])
    return distance

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python a-star.py [heuristic_type]")
        sys.exit(1)

    # Parse command-line argument for the heuristic type
    heuristic_type = int(sys.argv[1])

    # Read the initial state from standard input (stdin)
    init = sys.stdin.read().splitlines()
    init_state = [list(map(int, line.split())) for line in init]

    for i in range(3):
        for j in range(3):
            if init_state[i][j] == 0:
                row_index = i
                col_index = j
    
    # Create the initial state object and populate it with the parsed values
    initial_state = state()
    initial_state.xpos = row_index
    initial_state.ypos = col_index
    initial_state.tiles = init_state



    # Define your goal state
    goal_state = state()  # Set your goal state




    
    # Define a dictionary of heuristic functions based on heuristic_type
    heuristic_functions = {
        0: lambda state, goal: 0,
        1: lambda state, goal: sum(1 for i in range(3) for j in range(3) if state.tiles[i][j] != goal.tiles[i][j]),
        2: lambda state, goal: manhattan_distance(state, goal),
        3: lambda state, goal: column_distance_heuristic(state, goal),
        # Define other heuristics as needed
    }

    # Ensure the heuristic_type is valid
    if heuristic_type not in heuristic_functions:
        print("Invalid heuristic type.")
        sys.exit(1)

    # Run A* search with the specified heuristic and goal state
    result = a_star_algo(initial_state, goal_state, heuristic_functions[heuristic_type])

    depth = len(result['path']) - 1  # Subtract 1 to get the depth
    b = (result['nodes_visited'] ** (1 / depth)) if depth > 0 else 0.0

    # Print the results for this heuristic
    print(f"V: {result['nodes_visited']}")
    print(f"N: {result['max_nodes_in_memory']}")
    print(f"d: {depth}")
    print(f"b: {b}\n")

    for x in result['path']:
        print(x)
    

if __name__ == "__main__":
    main()
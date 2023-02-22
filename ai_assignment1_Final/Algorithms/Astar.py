import copy
import time
import sys
from abc import ABC
from typing import Tuple, Union, Dict, List, Any
import math
import numpy as np

from commonroad.scenario.trajectory import State

sys.path.append('../')
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node, CostNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import FIFOQueue, LIFOQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization

class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for search motion planners.
    """

    # declaration of class variables
    path_fig: Union[str, None]

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

    def initialize_search(self, time_pause, cost=True):
        """
        initializes the visualizer
        returns the initial node
        """
        self.list_status_nodes = []
        self.dict_node_status: Dict[int, Tuple] = {}
        self.time_pause = time_pause
        self.visited_nodes = []

        # first node
        if cost:
            node_initial = CostNode(list_paths=[[self.state_initial]],
                                        list_primitives=[self.motion_primitive_initial],
                                        depth_tree=0, cost=0)
        else:
            node_initial = Node(list_paths=[[self.state_initial]],
                                list_primitives=[self.motion_primitive_initial],
                                depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)
        self.dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                     status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(self.list_status_nodes), time_pause=self.time_pause)
        self.list_status_nodes.append(copy.copy(self.dict_node_status))
        return node_initial

    def take_step(self, successor, node_current, cost=True):
        """
        Visualizes the step of a successor and checks if it collides with either an obstacle or a boundary
        child cost is equal to the cost function up until this node
        Returns collision boolean and the child node if it does not collide
        """
        # translate and rotate motion primitive to current position
        list_primitives_current = copy.copy(node_current.list_primitives)
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])
        list_primitives_current.append(successor)
        self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
        if cost:
            child = CostNode(list_paths=self.path_new,
                                 list_primitives=list_primitives_current,
                                 depth_tree=node_current.depth_tree + 1,
                                 cost=self.cost_function(node_current))
        else:
            child = Node(list_paths=self.path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)

        # check for collision, skip if is not collision-free
        if not self.is_collision_free(path_translated):

            position = self.path_new[-1][-1].position.tolist()
            self.list_status_nodes, self.dict_node_status, self.visited_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                           path_translated=path_translated,
                                                                                           node_status=self.dict_node_status,
                                                                                           list_states_nodes=self.list_status_nodes,
                                                                                           time_pause=self.time_pause,
                                                                                           visited_nodes=self.visited_nodes)
            return True, child
        self.update_visuals()
        return False, child

    def update_visuals(self):
        """
        Visualizes a step on plot
        """
        position = self.path_new[-1][-1].position.tolist()
        if position not in self.visited_nodes:
            self.dict_node_status = update_visualization(primitive=self.path_new[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=self.dict_node_status, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(self.list_status_nodes), time_pause=self.time_pause)
            self.list_status_nodes.append(copy.copy(self.dict_node_status))
        self.visited_nodes.append(position)

    def goal_reached(self, successor, node_current):
        """
        Checks if the goal is reached.
        Returns True/False if goal is reached
        """
        path_translated = self.translate_primitive_to_current_state(successor,
                                                                    node_current.list_paths[-1])

        finalPath = []
        # goal test
        if self.reached_goal(path_translated):
            # goal reached

            self.path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            #print(node_current.list_paths[-1][-1])
            path_solution = self.remove_states_behind_goal(self.path_new)
            self.list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=self.dict_node_status,
                                                        list_states_nodes=self.list_status_nodes, time_pause=self.time_pause)

            #Get our final Path
            finalPathStates = []
            #For all States in Final Node's Path
            for n in path_solution:
                if n[0] not in finalPathStates:
                    finalPathStates.append(n[0])
            #For every state, return [x,y] positions
            for state in finalPathStates:
                finalPath.append([state.position[0], state.position[1]])

            #print(finalPath)
            return True,finalPath
        return False,finalPath

    def get_obstacles_information(self):
        """
        Information regarding the obstacles.
        Returns a list of obstacles' information, each element
        contains information regarding an obstacle:
        [x_center_position, y_center_position, length, width]

        """
        return self.extract_collision_obstacles_information()

    def get_goal_information(self):
        """
        Information regarding the goal.
        Returns a list of the goal's information
        with the following form:
        [x_center_position, y_center_position, length, width]
        """
        return self.extract_goal_information()

    def get_node_information(self, node_current):
        """
        Information regarding the input node_current.
        Returns a list of the node's information
        with the following form:
        [x_center_position, y_center_position]
        """
        return node_current.get_position()

    def cost_function(self, node_current):
        """
        Returns g(n) from initial to current node, !only works with cost nodes!
        """



        velocity = node_current.list_paths[-1][-1].velocity

        node_center = self.get_node_information(node_current)
        goal_center = self.get_goal_information()
        distance_x = goal_center[0] - node_center[0]
        distance_y = goal_center[1] - node_center[1]
        length_goal = goal_center[2]
        width_goal = goal_center[3]

        distance = 4.5
        if(abs(distance_x)<length_goal/2 and abs(distance_y)<width_goal/2):
            prev_x = node_current.list_paths[-2][-1].position[0]
            prev_y = node_current.list_paths[-2][-1].position[1]
            distance = goal_center[0] - length_goal / 2 - prev_x
            print("previous x= ", prev_x, "\t distance= ", distance)
       
        cost = node_current.cost + distance #* self.scenario.dt

        return cost

    def heuristic_function1(self, node_current):
        """
        Enter your heuristic function h(x) calculation of distance from node_current to goal
        Returns the distance normalized to be comparable with cost function measurements
        """
        # Euclidean Distance Heuristic
        [Xcurr, Ycurr] = self.get_node_information(node_current)
        [Xgoal, Ygoal, Lgoal, Wgoal] = self.get_goal_information()

        distance = math.sqrt( (Xcurr-Xgoal)**2 + (Ycurr-Ygoal)**2  )

        return distance



    def heuristic_function2(self, node_current):
        """
              Enter your heuristic function h(x) calculation of distance from node_current to goal
              Returns the distance normalized to be comparable with cost function measurements
              """
        # Manhattan Distance Heuristic
        [Xcurr, Ycurr] = self.get_node_information(node_current)
        [Xgoal, Ygoal, Lgoal, Wgoal] = self.get_goal_information()

        # print("###############################")
        # print(abs(Xcurr - Xgoal))
        # print("###############################")

        distance = abs(Xgoal-Xcurr) + abs(Ygoal-Ycurr)

        return distance



    def heuristic_function3(self, node_current):

        # Custom Distance Heuristic
        [Xcurr, Ycurr] = self.get_node_information(node_current)
        [Xgoal, Ygoal, Lgoal, Wgoal] = self.get_goal_information()


        obst=self.get_obstacles_information()
        sum=0

        distance=0;
        ###Closest Obstacle Detection
        minDist=math.inf
        for i in range(0,obst.__len__()):
            [x, y, length, width] = obst[i]
            Xobj=x
            if(Xcurr>x or Ycurr>y+width/2):
                sum=sum+1
                continue
            else:
                if(abs(x-Xcurr)<minDist):
                    minDist=abs(x-Xcurr)
                    Xobj=x
                    Yobj=y
                    LenObj=length
                    WidthObj=width


        ##If we're past all obstacles take Manhattan Distance to goal
        if(sum==obst.__len__()):
                distance = abs(Xgoal - Xcurr) + abs(Ygoal - Ycurr)
        else:
            w = WidthObj / 2
            ##If we're on case A (Below Goal)
            if (Ycurr - Ygoal <= 0):
                b = Xgoal - Xcurr
                ##If we can't or shouldn't choose this obstacle edge
                if((Yobj + w) >= 10 or abs(Ygoal-(Yobj+w)) > abs(Ygoal-(Yobj+w))):
                    a=Ycurr - (Yobj - w)
                    c = math.sqrt(b ** 2 + (abs(Yobj - w) - Ycurr) ** 2)
                else:
                    a = (Yobj + w) - Ycurr
                    c = math.sqrt(b ** 2 + (abs(Yobj + w) - Ycurr) ** 2)
            ##If we're on case B (Above Goal)
            if( (Ycurr-Ygoal>0)):
                b = Xgoal - Xcurr
                ##If we can't or shouldn't choose this obstacle edge
                if((Yobj-w)<=-10 or abs(Ygoal-(Yobj+w)) < abs(Ygoal-(Yobj+w))):
                    a = (Yobj + w) - Ycurr
                    c = math.sqrt(b ** 2 + (abs(Yobj + w) - Ycurr) ** 2)
                else:
                    a = Ycurr - (Yobj - w)
                    c = math.sqrt(b**2 + ( abs(Yobj-w) - Ycurr)**2)

            distance =  math.sqrt(a**2+(c ** 2 + a ** 2) )



        return distance





    def evaluation_function(self, node_current, w,heuristic):
        """
        f(x) = g(x) + h(x)
        """


        g = self.cost_function(node_current)

        if(heuristic==1):
            h = self.heuristic_function1(node_current)
        elif(heuristic==2):
            h = self.heuristic_function2(node_current)
        elif(heuristic==3):
            h = self.heuristic_function3(node_current)



        f =g + w*h
        # print(f)
        return f


    def goal_reached_wNode(self, node):
        [Xcurr, Ycurr] = self.get_node_information(node)
        [Xgoal, Ygoal, Lgoal, Wgoal] = self.get_goal_information()
        if Xcurr<= Xgoal + Lgoal/2 and Xcurr >= Xgoal - Lgoal/2 and Ycurr <= Ygoal + Wgoal/2 and Ycurr >= Ygoal - Wgoal/2:
            return True
        else:
            return False

    def returnBestChild(self, fringe, w,heuristic):
        minCost = math.inf

        for node in fringe:

            if self.evaluation_function(node, w,heuristic) < minCost:
                minCost = self.evaluation_function(node, w,heuristic)
                child = node

        return child

    def AstarSearch(self, current_node, fringe, w,heuristic, finalPath):
        #end if goad reached
        # goal_flag = self.goal_reached_wNode(current_node)
        #
        # if goal_flag:
        #     return True

        ##turn successors to Cost Nodes and Add them to fringe
        for successor in current_node.get_successors():

            collisionFlag, child = self.take_step(successor, current_node)

            goal_flag, self.finalPath = self.goal_reached(successor, current_node)

            if goal_flag:
                #print(finalPath)
                self.finalPath.append(self.get_node_information(child))
                return True, self.finalPath

            if not collisionFlag:
                fringe.append(child)

        ##remove current_node from fringe to find the next best child to continue
        fringe.remove(current_node)

        ###Choose best node to continue to
        bestChild = self.returnBestChild(fringe, w,heuristic)

        goal_found = self.AstarSearch(bestChild, fringe, w,heuristic, finalPath)

        if goal_found:
            return True, self.finalPath

        return False, self.finalPath




    def execute_search(self, w,heuristic, finalPath, time_pause) -> Tuple[Union[None, List[List[State]]], Union[None, List[MotionPrimitive]], Any]:
        node_initial = self.initialize_search(time_pause=time_pause)
        # print(self.get_obstacles_information())
        # print(self.get_goal_information())
        # print(self.get_node_information(node_initial))
        """Enter your code here"""
        ## Initializing vars
        fringe = []
        fringe.append(node_initial)
        result, finalPath = self.AstarSearch(node_initial, fringe,w,heuristic, [])

        visitedNodes = len(self.visited_nodes)
        heuristicFromStart = self.heuristic_function3(node_initial)

        finalNodeCost = finalPath[-1][0] - finalPath[0][0]


        return True, finalPath, heuristicFromStart,visitedNodes, finalNodeCost

class Astar(SequentialSearch):
    """
    Class for Astar Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)


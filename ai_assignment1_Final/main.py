import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    mpl.use('Qt5Agg')
except ImportError:
    mpl.use('TkAgg')

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# add current directory to python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.motion_planner import MotionPlanner
from SMP.motion_planner.plot_config import StudentScriptPlotConfig




def main():
    sys.setrecursionlimit(10000)
    f = open("output.txt", "w")
    args = sys.argv[1:]
    ## args = [scenario heuristic w_min w_max]
    # configurations
    w=0
    i = int(args[0])
    w_min = int(args[2])
    w_max = int(args[3])
    path_scenario = 'Scenarios/scenario%d.xml' % (i)
    file_motion_primitives = 'V_9.0_9.0_Vstep_0_SA_-0.2_0.2_SAstep_0.4_T_0.5_Model_BMW320i.xml'
    config_plot = StudentScriptPlotConfig(DO_PLOT=True)

    # load scenario and planning problem set
    scenario, planning_problem_set = CommonRoadFileReader(path_scenario).open()
    # retrieve the first planning problem
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

    # create maneuver automaton and planning problem
    automaton = ManeuverAutomaton.generate_automaton(file_motion_primitives)

    # comment out the planners which you don't want to execute
    dict_motion_planners = {
        #0: (MotionPlanner.DepthFirstSearch, "Depth First Search"),
        1: (MotionPlanner.Astar, "A*"),
        2: (MotionPlanner.IterativeDeepeningAstar, "IDA*")
    }

    f = open("output.txt", "a")
    scenarioName = '<Scenario %d>' % (i)
    f.write("=================================================\n")
    f.write(scenarioName + "\n")


    #f.write("heuristic " + args[1] + ":\n")
    for (class_planner, name_planner) in dict_motion_planners.values():
        planner = class_planner(scenario=scenario, planning_problem=planning_problem,
                                automaton=automaton, plot_config=config_plot)
        if i!=1:
            f.write("\n")

        if (name_planner == "A*"):

            for w in range(w_min,w_max+1):
                finalPath= []



                f.write(name_planner + "(w=" + repr(w) + "):\n")

                goalFound, found_path, heuristicFromStart, visitedNodes, finalNodeCost = planner.execute_search(w,int(args[1]),finalPath,time_pause=0.01)
                # lenPath=len(found_path)
                totalPath='\tFound path : '
                f.write("\tVisited Nodes :" + repr(visitedNodes) + "\n")
                f.write(totalPath)
                for node in found_path:
                    if node!=found_path[-1]:
                        f.write(repr(node) + "->" )
                    else:
                        f.write(repr(node)+"\n")


                f.write("\tHeuristic Cost(initial node) : " + repr(heuristicFromStart)+ "\n")
                f.write("\tEstimated Cost : " + repr(finalNodeCost)+ "\n")
        else:
            finalPath = []

            f.write(name_planner + "\n")

            goalFound, found_path, heuristicFromStart, visitedNodes, finalNodeCost = planner.execute_search(w, int(
                args[1]), finalPath, time_pause=0.01)
            # lenPath=len(found_path)
            totalPath = '\tFound path : '
            f.write("\tVisited Nodes :" + repr(visitedNodes) + "\n")
            f.write(totalPath)
            for node in found_path:
                if node != found_path[-1]:
                    f.write(repr(node) + "->")
                else:
                    f.write(repr(node) + "\n")

            f.write("\tHeuristic Cost(initial node) : " + repr(heuristicFromStart) + "\n")
            f.write("\tEstimated Cost : " + repr(finalNodeCost) + "\n")

    print('Done')


if __name__ == '__main__':
    main()


import argparse
import json
import random
import time

import pandas as pd
from accountant import Accountant
from drunkard import Drunkard
from dungeon_simulator import DungeonSimulator
from gambler import Gambler
from gambler_hw import GamblerHW


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='GAMBLER', help='Which agent to use')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='How quickly the algorithm tries to learn')
    parser.add_argument('--discount', type=float, default=0.98, help='Discount for estimated future action')
    parser.add_argument('--iterations', type=int, default=2000, help='Iteration count')
    parser.add_argument('--seed', type=int, default=42, help='For reproducibility')
    parser.add_argument('--exploration_rate', type=float, default=1.0, help='How much will gambler explore')
    parser.add_argument('--save_tag', type=str, default='', help='Additional tag to title for renaming')
    

    ARGS, unparsed = parser.parse_known_args()

    random.seed(ARGS.seed)

    # select agent
    if ARGS.agent == 'GAMBLER':
        agent = Gambler(learning_rate=ARGS.learning_rate, discount=ARGS.discount, iterations=ARGS.iterations)
    elif ARGS.agent == 'ACCOUNTANT':
        agent = Accountant()
    elif ARGS.agent == 'GAMBLER_HW':
        agent = GamblerHW(learning_rate=ARGS.learning_rate, discount=ARGS.discount, exploration_rate=ARGS.exploration_rate, iterations=ARGS.iterations)
    else:
        agent = Drunkard()

    if ARGS.agent == 'GAMBLER_HW':
        step_max = 5
    else:
        step_max = 250

    # setup simulation
    dungeon = DungeonSimulator()
    dungeon.reset()
    total_reward = 0 # Score keeping
    last_total = 0

    save_data = []

    # main loop
    for step in range(ARGS.iterations):
        old_state = dungeon.state # Store current state
        action = agent.get_next_action(old_state) # Query agent for the next action
        new_state, reward = dungeon.take_action(action) # Take action, get new state and reward
        agent.update(old_state, new_state, action, reward) # Let the agent update internals

        total_reward += reward # Keep score
        if step % step_max == 0: # Print out metadata every 250th iteration
            performance = (total_reward - last_total) / 250.0
            print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
            last_total = total_reward
            save_data.append([step,total_reward])

        time.sleep(0.0001) # Avoid spamming stdout too fast!
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("\n")
    print("Final Q-table\n", pd.DataFrame(agent.q_table))
    pd.DataFrame(save_data).to_csv('simple_RL_'+ARGS.agent+ARGS.save_tag+'.csv', header=['Step','Total Reward'],index=False)

if __name__ == "__main__":
    main()

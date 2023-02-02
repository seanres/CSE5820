import random
import json
import argparse
import time
from drunkard import Drunkard
from accountant import Accountant
from gambler import Gambler
from dungeon_simulator import DungeonSimulator
import pandas as pd

def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='GAMBLER', help='Which agent to use')
    parser.add_argument('--learning_rate', type=float, default=0.5, help='How quickly the algorithm tries to learn')
    parser.add_argument('--discount', type=float, default=0.98, help='Discount for estimated future action')
    parser.add_argument('--iterations', type=int, default=2000, help='Iteration count')
    parser.add_argument('--seed', type=int, default=42, help='For reproducibility')
    

    ARGS, unparsed = parser.parse_known_args()

    random.seed(ARGS.seed)

    # select agent
    if ARGS.agent == 'GAMBLER':
        agent = Gambler(learning_rate=ARGS.learning_rate, discount=ARGS.discount, iterations=ARGS.iterations)
        print('gambler')
    elif ARGS.agent == 'ACCOUNTANT':
        agent = Accountant()
    else:
        agent = Drunkard()
        print('picked the drunk!!', ARGS.agent)



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
        if step % 250 == 0: # Print out metadata every 250th iteration
            performance = (total_reward - last_total) / 250.0
            print(json.dumps({'step': step, 'performance': performance, 'total_reward': total_reward}))
            last_total = total_reward
            save_data.append([step,total_reward])

        time.sleep(0.0001) # Avoid spamming stdout too fast!
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print("\n")
    print("Final Q-table\n", pd.DataFrame(agent.q_table))
    pd.DataFrame(save_data).to_csv('simple_RL_'+ARGS.agent+'.csv', header=['Step','Total Reward'],index=False)

if __name__ == "__main__":
    main()

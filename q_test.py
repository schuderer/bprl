# Test code (simple q learner (discretized state & action space)):

import sys
import random
import numpy as np
import gym
import pension_env as penv
import agent
import value_function
# from utils import do_profile
import logging

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)


# @do_profile(follow=[agent.Agent.run_episode,
#                     value_function.ActionValueFunction.select_action,
#                     value_function.ActionValueFunction.update_value,
#                     agent.greedy,
#                     agent.epsilon_greedy])
def learn(agent, episodes, max_steps):
    overall = 0
    last_100 = np.zeros((100,))
    num_actions = agent.q_function.action_disc.space.n
    for episode in range(episodes):
        logger.debug('size of q table: %s', len(agent.q_function.q_table.keys())*num_actions)

        q_table, cumul_reward, num_steps, info = \
            agent.run_episode(max_steps=max_steps, exploit=False)

        overall += cumul_reward
        last_100[episode % 100] = cumul_reward
        logger.warning('Episode %s finished after %s timesteps with cumulative reward %s (last 100 mean = %s)',
                       episode, num_steps, cumul_reward, last_100.mean())
        if type(agent.env).__name__ == 'PensionEnv':
            logger.warning('year %s, q table size %s, epsilon %s, alpha %s, #humans %s, reputation %s',
                        agent.env.year, len(q_table.keys()) * num_actions,
                        agent.epsilon, agent.alpha,
                        len([h for h in agent.env.humans if h.active]),
                        info['company'].reputation)
        else:
            logger.warning('q table size %s, epsilon %s, alpha %s',
                        len(q_table.keys()), agent.epsilon, agent.alpha)

    logger.warning('Overall cumulative reward: %s', overall/episodes)
    logger.warning('Average reward last 100 episodes: %s', last_100.mean())
    return q_table


# Run Q-Learning

# env = penv.PensionEnv()
# num_bins = 12
# log_bins = True

# env = gym.make('Pendulum-v0')
# num_bins = 10
# log_bins = True

# env = gym.make('CartPole-v0')
# num_bins = 10
# log_bins = False

env = gym.make('MountainCar-v0')
num_bins = 20
log_bins = False

# env = gym.make('FrozenLake-v0')
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name': '4x4', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )
# env = gym.make('FrozenLakeNotSlippery-v0')
# num_bins = 1
# log_bins = False


logger.info('###### LEARNING: ######')

logger.setLevel(logging.WARNING)
penv.logger.setLevel(logging.WARNING)

env.investing = False

seed = 7

# TODO doesn't work somehow for envs
env.seed(seed)  # environment can have its own seed

# seeds for agent
random.seed(seed)
np.random.seed(seed)

q_func = value_function.ActionValueFunction(env,
                         default_value=0,
                         discretize_bins=num_bins,
                         discretize_log=log_bins)

agent = agent.Agent(env,
                 q_function=q_func,
                 update_policy=agent.greedy,
                 exploration_policy=agent.epsilon_greedy,
                 gamma=0.99,
                 min_alpha=0.1,
                 min_epsilon=0.1,
                 alpha_decay=1,   # default 1 = fixed alpha (min_alpha)
                 epsilon_decay=1  # default: 1 = fixed epsilon (instant decay to min_epsilon)
                 )

q_table = learn(agent, episodes=1000, max_steps=20000)


logger.info('###### TESTING: ######')

logger.setLevel(logging.INFO)

for _ in range(3):
    reward = agent.run_episode(exploit=True)[1]
    logger.info("reward: %s", reward)



# logger.info('###### LEARNING: ######')
#
# logger.setLevel(logging.WARNING)
# penv.logger.setLevel(logging.WARNING)
#
# env.investing = False
#
# seed = 7
#
# # TODO doesn't work somehow for envs
# env.seed(seed)  # environment can have its own seed
#
# # seeds for agent
# random.seed(seed)
# np.random.seed(seed)
#
# agent = Agent(env,
#               q_table={},
#               gamma=0.99,
#               min_alpha=0.1,    # was: 0.01
#               min_epsilon=0.1,  # was: 0.03
#               alpha_decay=1,    # default 1 = fixed alpha (instant decay to min_alpha)
#               epsilon_decay=1,  # default: 1 = fixed epsilon (instant decay to min_epsilon)
#               default_value=0,
#               discretize_bins=num_bins,
#               discretize_log=log_bins
#               )
#
# q_table = learn(agent, episodes=1000, max_steps=20000)
#
#
# logger.info('###### TESTING: ######')
#
# logger.setLevel(logging.INFO)
#
# for _ in range(3):
#     reward = agent.run_episode(exploit=True)[1]
#     logger.info("reward: %s", reward)


# qTable = q_learn(env,
#                  min_alpha=0.01,     # temperature/learning rate, was 0.01
#                  alpha_decay=1,      # reduction factor per episode, was 1
#                  gamma=0.99,         # discount factor, was 0.99
#                  min_epsilon=0.03,   # minimal epsilon (exploration rate for e-greedy policy), was 0.03
#                  epsilon_decay=1,    # reduction per episode, was 1
#                  episodes=15000,     # was: 15000
#                  max_steps=20000,     # abort episode after this number of steps, was: 20000
#                  q_table={},
#                  average_rewards=False)


# print_q2(qTable)
#
#
# exit()

# qTable_long = {'8-11-0': 94.85750359596473, '9-11-1': 93.40323065017274, '8-10-0': 54.05487303099865, '9-10-0': 69.4452590372413, '9-6-0': 28.350606637340828, '10-10-1': 0.0, '8-11-1': 88.69965983586648, '9-10-1': 0.0, '9-9-0': 43.48613366033944, '9-11-0': 95.0580167838315, '8-6-0': 2.9663781106297673, '8-10-1': 0.0, '0-10-1': 15.319100097519744, '10-9-1': 0.0, '8-9-1': 0.0, '10-11-1': 95.5552419744198, '0-11-0': 74.40527194908847, '9-6-1': 0.0, '10-10-0': 77.7631753592817, '9-9-1': 0.0, '8-9-0': 10.466543791570732, '10-11-0': 96.16730642224013, '10-9-0': 52.50022631368486, '10-6-1': 0.0, '10-6-0': 47.267848204387754, '8-12-1': 96.89479613623092, '9-12-0': 97.69003889839313, '10-12-1': 97.72497266156432, '8-12-0': 98.29330602203332, '9-12-1': 97.66739364403784, '11-11-0': 97.39174531384064, '11-10-0': 90.39064853287226, '11-12-0': 98.44059070537676, '11-11-1': 96.537239260211, '12-12-1': 98.20931866611063, '12-11-1': 95.9104943675021, '11-9-1': 0.0, '10-12-0': 98.30945416910251, '11-10-1': 0.0, '12-10-1': 0.0, '11-12-1': 98.08502073963916, '12-11-0': 96.45089471340923, '11-6-0': 25.099394364670093, '12-10-0': 91.85700280656347, '12-9-0': 49.73730421259588, '12-6-0': 24.967896842942707, '12-12-0': 97.48316451537686, '11-9-0': 49.55524985337189, '12-9-1': 0.0, '0-11-1': 74.09507814228127, '0-6-1': 0.0, '0-9-1': 0.3472440423522913, '0-9-0': 1.0870036388169635, '0-10-0': 13.289073071107142, '0-6-0': 0.17096349741069655, '0-12-1': 81.01054815466709, '8-6-1': 0.0, '10-13-1': 62.000330888305314, '9-13-1': 65.37576691088769, '8-13-1': 21.784954532010495, '11-13-0': 0.7057965640065662, '10-13-0': 0.9732023924806424, '0-12-0': 19.786256147889635, '11-13-1': 34.36227034336228, '9-13-0': 0.43486906879583603, '8-13-0': 0.01, '12-13-1': 0.9807091303179641}


#CartPoleEnv
# qTable_cartpole = {'5-5-5-4-0': 65.79490669031206, '5-4-5-5-1': 64.57569278848801, '5-5-5-5-1': 68.49154571786907, '5-5-5-4-1': 67.78739136587991, '5-5-4-4-1': 56.32032629459533, '5-5-4-4-0': 60.351318229349104, '5-5-3-4-1': 26.412916541887174, '5-5-3-4-0': 46.137147703313836, '5-4-4-5-0': 72.06071849192973, '5-4-4-5-1': 72.11543487449585, '4-4-5-5-0': 67.83058928464905, '4-4-5-5-1': 76.66090494844029, '4-5-5-5-1': 76.06487578430338, '4-5-5-4-0': 76.33710584431064, '4-5-5-4-1': 74.80620239157572, '4-5-5-5-0': 75.60527941838072, '4-4-6-5-0': 39.83443593592824, '4-4-6-5-1': 75.77655006878584, '5-5-4-5-0': 73.95532194368253, '5-4-5-5-0': 60.52180088641191, '4-4-4-5-0': 77.0086911880735, '4-4-4-5-1': 73.40541594237541, '4-4-5-4-1': 76.90526572006586, '4-5-4-4-1': 58.03854228502024, '5-4-4-4-1': 76.58156035541451, '4-4-5-4-0': 77.07703652143826, '4-4-4-4-0': 72.71897012996415, '4-5-4-5-1': 66.87072833790249, '4-5-4-4-0': 71.12586244934516, '4-5-6-5-0': 66.89856933267598, '5-5-5-5-0': 68.06105097418555, '5-4-6-5-1': 43.00838649478901, '5-5-6-5-0': 46.83489408845276, '5-4-6-5-0': 29.249089041239273, '5-4-5-4-1': 82.10537185010458, '4-5-6-4-1': 76.47937167101995, '4-5-6-5-1': 77.42371738828543, '4-5-6-4-0': 72.99033204637685, '5-5-6-5-1': 67.83226450857303, '5-4-4-4-0': 76.83772635801124, '4-5-3-4-0': 43.199314383183214, '4-5-3-4-1': 33.62985051262434, '5-4-3-4-1': 64.73454521766631, '5-4-3-4-0': 83.12305595819369, '5-4-3-5-0': 77.86899659168468, '4-4-3-5-0': 75.98653411722296, '4-4-3-5-1': 60.96725461273007, '4-4-4-4-1': 69.41254547511306, '4-4-3-4-1': 50.64212968220821, '4-4-3-4-0': 62.03932900328076, '5-4-5-4-0': 65.00642910135853, '5-4-3-5-1': 71.02257203656347, '5-5-6-4-1': 70.50523002146409, '5-5-6-4-0': 64.90548269885909, '5-5-4-5-1': 73.37858483628298, '4-5-4-5-0': 77.8673975515054, '5-5-3-5-0': 82.45839347217455, '6-5-4-4-0': 27.795285427749988, '6-5-3-4-0': 28.66777234182507, '6-5-3-5-1': 26.30055116826335, '6-5-3-4-1': 16.693431314178405, '4-4-6-4-0': 67.59513442199535, '5-5-3-5-1': 67.09515847301057, '6-5-6-4-0': 26.022973646916576, '6-5-6-5-0': 18.228118807338774, '6-5-6-5-1': 28.610263209067483, '6-5-5-4-0': 23.615101321475862, '6-5-5-4-1': 29.62433480415893, '6-5-4-4-1': 19.53766386458999, '6-5-3-5-0': 33.43744281453817, '6-5-4-5-0': 31.27993374862174, '6-5-4-5-1': 25.551840889590054, '6-4-5-5-0': 6.516933012471345, '6-4-6-5-0': 2.1531633236487546, '6-5-5-5-1': 30.04097524280532, '6-5-5-5-0': 26.80842708610143, '3-4-3-4-0': 24.32064801155313, '3-4-4-4-0': 27.808597167353916, '3-4-4-5-1': 27.712150555576518, '3-4-4-4-1': 24.772981154194735, '3-4-3-5-1': 22.059325391412795, '3-4-3-4-1': 16.46366089493981, '4-4-6-4-1': 83.97352476432845, '3-4-4-5-0': 24.584782721014022, '6-4-4-5-0': 8.54251041118917, '6-4-4-5-1': 31.4440417651805, '6-4-6-5-1': 14.91540432742624, '6-5-6-4-1': 29.44482970314337, '6-4-5-5-1': 28.932727160593593, '3-4-5-5-0': 12.52513350815326, '3-4-5-5-1': 12.463640200981548, '3-4-5-4-0': 20.679388743564207, '3-4-5-4-1': 25.97113176881964, '3-4-3-5-0': 27.405930935077382, '6-4-3-5-0': 23.89305078223371, '3-4-6-5-0': 3.7077858804422483, '3-4-6-5-1': 3.283656251135147, '3-5-3-4-1': 2.051892289417734, '3-5-4-4-0': 18.66532595315425, '5-4-6-4-0': 7.821258581459492, '3-4-6-4-0': 2.6716895481300553, '3-5-3-4-0': 0.403687570066275, '6-4-4-4-0': 4.332397326693339, '6-4-4-4-1': 2.2499377784411747, '3-5-5-4-0': 20.29009808436112, '4-5-3-5-0': 20.302462038711976, '3-5-5-5-0': 4.589863601732336}


# Mountain Car
# qTable = {'8-10-0': -71.68061905514621, '8-10-1': -74.92224928223072, '8-10-2': -74.9878820244465, '8-11-0': -74.84779056027044, '8-11-2': -69.82064759293628, '8-11-1': -74.6039546354059, '7-10-1': -71.70071404904247, '7-10-2': -71.63827999612792, '7-10-0': -71.7095492823004, '7-11-0': -71.71803767027124, '7-11-2': -70.94847347435643, '7-11-1': -71.70507395023277, '8-12-1': -67.6585726034916, '8-12-2': -67.07191613921802, '8-12-0': -68.55656844635465, '9-12-1': -68.32127339386336, '9-12-2': -65.46136639271039, '9-12-0': -68.34217171131411, '10-12-0': -63.15895549329581, '10-11-2': -65.4058204734137, '10-11-0': -63.377055396086924, '10-10-2': -65.88180647110809, '10-10-0': -62.50846262933951, '10-9-1': -62.91575709774243, '10-9-2': -63.71058586950611, '9-9-0': -69.3769506025905, '9-9-1': -70.80182468175734, '9-9-2': -70.8035093482799, '8-9-0': -69.93304733111434, '8-8-0': -68.72132668658065, '8-8-2': -69.12669850745269, '7-8-0': -68.37776964241262, '7-8-1': -69.05732538588563, '7-8-2': -68.87684924466117, '6-9-0': -67.54552927924524, '6-9-2': -67.53872383574458, '6-9-1': -67.96030408062033, '6-10-0': -68.0356477337898, '6-10-1': -68.02845177561731, '6-10-2': -67.02187284847133, '6-11-0': -67.0619761957927, '6-11-1': -64.95124288708735, '6-11-2': -67.36101008856238, '6-12-1': -65.63733588959221, '6-12-0': -65.40990437792063, '6-12-2': -64.40136352199337, '7-12-1': -67.68600664504558, '7-13-1': -66.14048586472262, '7-13-0': -66.67749766817174, '7-13-2': -63.40378158440743, '8-13-0': -66.84770793254698, '8-13-2': -65.1359680187345, '8-13-1': -66.80080816716357, '9-13-1': -65.57036800031966, '9-13-2': -65.22926437912672, '9-13-0': -65.66119992734143, '8-9-2': -70.67066454252762, '9-11-1': -73.38492266697706, '9-11-2': -73.38605516848334, '9-11-0': -73.4029873784656, '9-10-2': -73.36496276886325, '9-10-0': -70.70987537128086, '9-10-1': -73.21205455761388, '8-9-1': -70.86630662146278, '8-8-1': -68.77666590254935, '7-9-2': -70.57647019892768, '7-9-0': -69.09461319322327, '7-12-2': -66.52162297509477, '10-13-0': -63.58334008409851, '10-12-2': -66.28786095035855, '10-12-1': -65.57072522878754, '11-11-1': -61.98598182062503, '11-11-2': -61.6881692153532, '11-11-0': -59.94825574424165, '11-10-1': -60.26696744652156, '11-10-0': -58.707685224058324, '10-9-0': -60.81246461011113, '10-8-1': -58.44830096933444, '10-8-0': -59.39763011007253, '10-8-2': -60.97767132404075, '9-8-1': -63.61643967565367, '9-8-0': -57.61123217872495, '9-8-2': -63.675573609298006, '8-7-0': -55.26519020949829, '8-7-2': -60.97453209534845, '7-9-1': -70.05516518417204, '7-12-0': -67.77244736184919, '10-10-1': -64.43872083051177, '10-11-1': -64.99786271873099, '6-8-2': -67.42909438053626, '6-8-0': -67.56572005381001, '5-9-0': -66.1593687944244, '5-9-2': -65.82530932167799, '5-10-1': -65.85543605134447, '5-10-2': -65.5272216263444, '5-11-0': -65.67836259401727, '5-11-2': -65.18755961541763, '5-11-1': -64.88707653075134, '5-12-1': -65.24469040112588, '6-13-0': -63.82140888533905, '6-8-1': -67.5326148168834, '8-14-0': -65.36241458703729, '10-13-1': -63.89006214659097, '5-9-1': -66.1197046021834, '5-12-2': -63.49247654237829, '6-13-2': -63.521345409478045, '6-14-1': -63.22170837219955, '6-14-2': -59.00325561327856, '7-14-0': -64.17264572270985, '7-14-1': -63.395413813567664, '8-14-2': -63.00195305786202, '8-14-1': -64.91185414510925, '9-14-1': -64.78826766776444, '9-14-0': -65.2142301109137, '10-14-1': -62.94524117860803, '10-14-0': -63.85194971254936, '10-13-2': -63.849502181016106, '11-13-2': -61.68610761978225, '11-13-1': -61.765600267092076, '11-13-0': -61.2389655654853, '11-12-2': -61.783485089292, '11-12-0': -60.37288724651628, '11-12-1': -61.48346870492939, '12-11-1': -59.53533519563242, '12-10-1': -58.879526213762574, '12-10-0': -58.64668505238582, '11-9-2': -59.8689136036675, '11-9-1': -58.92740132614767, '11-9-0': -57.454284077111495, '11-8-1': -57.74923683250803, '10-7-0': -56.710259538596006, '10-7-2': -56.69082775984605, '9-7-2': -57.98472996201286, '8-7-1': -60.421443061292926, '7-7-0': -54.22392537691482, '6-7-0': -53.54894719362775, '6-7-1': -60.52316186279922, '5-7-2': -57.388198110336404, '5-8-2': -64.74815980924203, '5-8-1': -64.76991175566161, '5-8-0': -59.15424203835962, '4-9-0': -56.61204824646731, '4-9-2': -56.42739215440797, '4-10-2': -57.347450242529845, '4-11-0': -57.6420258107523, '4-11-1': -56.102768699865514, '4-11-2': -57.32611175054664, '4-12-2': -50.4109510844143, '5-12-0': -65.4143694250138, '5-13-2': -60.572689886367485, '5-13-1': -57.19160910703813, '5-13-0': -60.619845298014695, '6-14-0': -62.869281799305085, '9-14-2': -63.08058739807615, '10-14-2': -63.59797496730689, '11-8-0': -57.241721783773755, '10-7-1': -56.13779008714515, '9-7-0': -54.90587369535345, '7-7-2': -60.57295747166225, '7-7-1': -60.09265202915927, '6-7-2': -60.59542129745358, '11-10-2': -60.53754291796757, '9-7-1': -57.451825017563735, '5-10-0': -66.53409332264872, '6-13-1': -63.448145675358774, '7-14-2': -63.14500478580557, '5-7-1': -56.714184722505564, '4-10-0': -56.974321758190015, '4-10-1': -56.932453412302436, '12-11-0': -59.17102611001415, '7-6-2': -56.24604486409651, '7-15-0': -60.55759785374997, '8-15-1': -62.73287159578631, '8-15-0': -62.714965689077815, '12-11-2': -59.284230705593465, '11-8-2': -57.43785397222032, '4-9-1': -55.83203371598173, '9-15-2': -62.62049039024862, '9-15-0': -62.64898117508768, '12-13-2': -60.843652901427006, '12-12-1': -59.937698845892065, '12-12-2': -60.08237871618921, '12-10-2': -58.622653797574166, '12-9-0': -56.916924489115615, '12-9-2': -57.774828740555975, '12-8-2': -56.927486063770786, '8-6-2': -54.48447072775804, '7-6-0': -52.08605808512569, '8-6-0': -52.7606086323002, '5-7-0': -48.9516810907602, '4-8-1': -55.04297748201759, '4-12-1': -55.88942029945688, '4-12-0': -55.05642180384658, '4-13-2': -41.54985302361655, '7-15-1': -57.60513141976336, '7-15-2': -55.75943728132071, '8-15-2': -62.733132670675175, '11-14-2': -62.33293831113651, '11-14-0': -61.808212862766624, '12-13-0': -60.266961136319175, '12-13-1': -60.71909516700832, '12-12-0': -60.15736056771132, '12-9-1': -57.77044897417703, '12-8-0': -55.31173364436562, '9-6-1': -55.34919350203518, '9-6-2': -55.4162744123773, '11-14-1': -62.00167643601382, '11-7-1': -56.2681248326715, '9-6-0': -53.50149801006438, '8-6-1': -55.722258827133025, '6-6-1': -54.893503859877434, '6-6-0': -50.57340230373585, '5-6-0': -48.36046200537178, '4-6-1': -47.090844581821905, '4-6-2': -49.90684035567453, '4-7-2': -51.64442292395743, '3-8-1': -46.78294518565325, '3-8-0': -47.79856737131754, '3-9-1': -53.13892557250287, '3-9-0': -44.06846309671701, '3-10-2': -52.48658722873761, '3-10-1': -43.07572257263835, '3-11-1': -41.70297025825777, '3-11-2': -51.67940014696816, '3-12-1': -39.33096004520927, '3-12-0': -45.18793130358875, '3-13-2': -37.76093416174123, '3-13-0': -45.530056948058274, '7-6-1': -54.179329979455886, '4-8-2': -55.26314614982149, '4-13-1': -51.696105925616564, '5-14-0': -50.375410968969895, '9-15-1': -62.59586808213095, '10-15-2': -61.715967139983405, '10-15-0': -61.96956433330184, '11-15-1': -59.46455325827353, '11-7-0': -55.711534768404285, '5-14-1': -51.80539317317967, '11-7-2': -56.57907555522424, '6-6-2': -54.11880468812296, '5-6-2': -51.211052687724504, '4-7-0': -47.06286597992735, '4-7-1': -50.83052860704041, '4-8-0': -47.789931416114364, '3-9-2': -47.27036571968188, '3-10-0': -51.01083617555194, '3-11-0': -49.04100776447206, '3-12-2': -43.852127327845736, '13-12-1': -58.65830917207239, '13-11-1': -57.82830243267347, '13-11-2': -57.98453406239234, '13-11-0': -55.73670442218499, '13-10-0': -54.78629610637452, '13-10-2': -57.725948670711915, '10-6-2': -54.722760074721464, '5-14-2': -41.391016778191556, '6-15-0': -53.5367652133922, '10-15-1': -61.961960314208284, '7-5-1': -51.84301567121674, '5-6-1': -50.39457988285746, '3-8-2': -44.22656230538003, '3-13-1': -41.75942300959412, '4-14-0': -44.243503093737296, '4-14-2': -41.44652375889269, '5-15-1': -39.79283440137349, '5-15-2': -33.398035296901185, '6-15-1': -53.528703056250905, '7-16-1': -47.188303044657985, '7-16-0': -46.5482324473889, '8-16-0': -50.05046109386971, '8-16-2': -49.8329202485167, '8-16-1': -50.126145777559444, '10-6-0': -54.58259010911244, '11-15-0': -59.71137175644657, '12-8-1': -57.163980097157314, '4-13-0': -52.51689884257802, '12-14-0': -59.704920398076304, '4-6-0': -50.0328671324301, '3-7-1': -45.117251455059204, '4-14-1': -36.71033127307647, '10-6-1': -54.35076635009143, '9-5-1': -52.52702870257011, '8-5-2': -52.50468108320874, '5-15-0': -44.7231597502512, '6-15-2': -43.34618056764658, '12-14-1': -59.56930722095173, '13-12-0': -56.594789701363645, '13-10-1': -57.47977181828067, '13-9-0': -54.42532264718726, '11-15-2': -59.73576846364267, '12-14-2': -58.996411889503214, '13-13-2': -57.472363268887754, '13-13-0': -57.353391325538134, '13-8-0': -52.8239176970507, '12-7-0': -52.55994821070739, '12-7-2': -55.157430818088294, '11-6-0': -52.73077174516645, '11-6-2': -53.880078819743304, '11-6-1': -53.18708029727166, '13-9-2': -55.9917823084549, '10-5-1': -51.950942397726905, '9-5-2': -52.78846585210786, '8-5-1': -52.53104756464492, '8-5-0': -51.733838924993364, '6-5-1': -50.695069834542345, '9-16-2': -52.7148868881137, '9-16-1': -52.576401318071504, '12-15-2': -54.49137097909824, '13-9-1': -55.64637448335916, '13-8-2': -53.6795600006928, '13-8-1': -53.40664387154755, '12-7-1': -55.40082788827982, '10-5-0': -50.25683841157184, '9-5-0': -52.67276678445516, '7-5-0': -50.77718335940164, '6-5-0': -49.76618941619686, '5-5-0': -48.6042927028595, '5-5-2': -49.3182185462168, '3-7-2': -47.11172736111836, '9-16-0': -52.76405505142851, '13-14-0': -57.622219453893436, '13-12-2': -58.62183293010219, '8-4-0': -47.51903992281976, '7-4-0': -48.04166042617227, '7-4-1': -49.2056099008893, '6-4-1': -48.357526105620636, '6-4-0': -47.66656900957465, '5-4-2': -45.96013511053289, '5-5-1': -49.42353526993382, '4-5-0': -47.23900284709158, '4-5-1': -48.11413607822055, '3-6-0': -46.34242531550384, '2-6-0': -42.75619274792278, '2-7-0': -43.19841144606072, '1-7-1': -40.79377585227394, '1-7-2': -40.8136508929602, '1-8-0': -40.87864425913669, '1-8-2': -40.78279297533927, '1-10-0': -39.17036735653291, '1-11-1': -38.4822666416309, '1-11-0': -38.481536695762635, '1-12-1': -37.15930674800364, '1-12-0': -37.336942789823816, '1-12-2': -36.83453708742415, '1-13-0': -36.066916536559404, '10-16-1': -52.22674345171431, '6-16-1': -36.575700597585545, '10-16-2': -53.43873917426516, '10-16-0': -53.565890796691136, '12-15-0': -57.41953186873823, '13-13-1': -57.39427005313586, '3-7-0': -49.10903134732321, '2-7-2': -43.41091547790766, '2-8-0': -44.91527444379262, '1-9-0': -40.11987900359774, '1-9-1': -40.06206806644887, '1-9-2': -40.903012542590886, '2-8-1': -45.00141626266389, '2-9-1': -44.39712501697557, '2-9-0': -45.95031624537878, '2-9-2': -42.34903029093911, '2-10-0': -44.37204239286033, '2-10-1': -43.5627212596107, '2-11-1': -44.388823008979195, '2-11-2': -40.329436447084994, '13-14-1': -56.38507476233412, '14-12-0': -55.79529567514372, '14-11-2': -55.20103207055426, '14-11-0': -55.084483633268185, '14-10-0': -54.27490234428634, '14-10-2': -54.48334642484486, '14-9-1': -53.44787218516573, '12-6-2': -51.48116272946007, '10-5-2': -51.71699789619867, '7-5-2': -51.84247714199415, '6-5-2': -50.89451623070585, '12-6-0': -50.80883039225574, '11-16-1': -47.05089293997252, '11-16-0': -47.38536915542465, '2-11-0': -43.455423368489264, '2-12-0': -43.891416279864174, '4-15-0': -34.79108707233186, '5-16-1': -37.06412816931101, '6-16-0': -36.48600002607578, '7-16-2': -33.24045038066121, '14-11-1': -55.21758494674245, '8-4-2': -49.058805932552204, '2-8-2': -42.67205327690151, '2-10-2': -41.420768422804564, '14-10-1': -54.75533218943465, '3-14-0': -39.83162438082585, '4-15-1': -38.97332501429971, '13-14-2': -57.603021946222896, '14-12-2': -55.963498350407264, '14-12-1': -55.86369625258615, '14-9-2': -54.20283093214486, '6-16-2': -32.275831273162744, '12-15-1': -57.49237904722615, '14-13-0': -56.305945026712855, '14-9-0': -53.519960715246405, '13-7-2': -52.437534261610494, '12-6-1': -51.539822421986145, '11-5-0': -49.314560905561315, '10-4-0': -48.167398765503926, '9-4-2': -49.76240904941904, '2-12-1': -43.98080090752199, '3-14-2': -36.17233713642092, '5-16-0': -39.601268661958834, '7-17-0': -33.48114748331086, '8-17-0': -35.343155699208296, '9-17-0': -34.430770181633136, '11-16-2': -40.993406633361694, '13-15-0': -47.623341198378654, '14-13-2': -56.344034985149996, '14-13-1': -56.32782804243994, '14-8-1': -52.70049116045965, '13-7-0': -51.73721190010817, '12-5-0': -49.825177507789775, '9-4-1': -49.13103902217864, '9-3-1': -46.93405490880267, '8-3-0': -46.62768045080485, '7-3-0': -45.99411297475483, '6-3-0': -45.204057235606236, '6-3-2': -45.388776870291984, '5-3-0': -44.29871521805953, '5-4-0': -47.223133527331804, '4-4-0': -43.96655027312472, '3-4-0': -42.83973971987762, '3-4-2': -42.69452813867599, '2-5-0': -42.71879760937743, '1-5-0': -40.27529439561938, '1-6-0': -40.38657543898294, '12-16-0': -44.610205468437414, '12-16-2': -28.98537121681222, '14-14-0': -49.83346548508411, '15-12-0': -51.74261550070679, '15-11-0': -52.97558031101977, '15-10-1': -52.265017214626546, '15-10-0': -51.94536604388207, '14-8-0': -52.68504693414076, '2-12-2': -39.375899457198415, '4-15-2': -38.70516679118985, '9-17-2': -30.227504846733083, '10-17-0': -38.48898783472573, '2-13-2': -37.13608914178683, '8-17-2': -30.391756234894103, '9-4-0': -47.58643755747963, '8-4-1': -49.06023551031963, '7-4-2': -49.23090034415528, '5-4-1': -47.28759196243818, '4-5-2': -47.84170250123917, '2-6-2': -42.78934524664988, '1-10-2': -39.07741753671799, '13-6-0': -50.556339241662855, '10-4-1': -48.87881256370334, '3-5-0': -44.63380084634144, '3-5-1': -44.61788042385507, '3-6-1': -45.575172480049595, '2-6-1': -42.21304348781059, '3-6-2': -46.38956581312578, '1-10-1': -39.15030737514307, '1-11-2': -38.245092663888215, '2-13-1': -38.992717610145746, '2-13-0': -38.55180603490105, '2-14-0': -34.90293262744276, '3-14-1': -39.607550079560944, '3-15-0': -34.63655683598448, '11-5-1': -50.03236442684856, '10-4-2': -49.29641000333126, '13-15-1': -45.152044280255105, '13-7-1': -52.47487619398099, '8-17-1': -38.6650005285139, '10-17-2': -29.92081243992675, '14-14-1': -42.34395886297243, '15-9-0': -49.8593375729741, '9-17-1': -38.13742524281071, '13-16-0': -30.707487907061704, '14-15-0': -40.099957375590954, '15-13-0': -48.146892196726284, '15-13-1': -48.877975001630716, '15-12-1': -53.35146736173319, '12-16-1': -44.5732226467586, '14-14-2': -50.26532051281531, '15-11-2': -53.181309173969495, '4-16-0': -32.00881684589964, '5-16-2': -32.58463908419849, '6-4-2': -48.638046313266, '6-17-1': -36.70839463920962, '6-17-0': -37.41930228528888, '11-5-2': -50.27304239685255, '2-7-1': -43.79272043494509, '4-16-1': -31.89410849995131, '7-17-1': -36.46595772113663, '14-8-2': -52.66912559380291, '15-11-1': -51.72088309583771, '14-7-0': -49.6459270135168, '13-6-2': -50.623154520852125, '10-17-1': -37.87785255628125, '4-4-2': -45.00687767397762, '11-17-2': -36.34415032482662, '11-17-0': -37.45133992419059, '13-15-2': -34.04857384998824, '15-9-1': -51.04938066308247, '15-8-1': -49.109509066124886, '1-7-0': -40.82344521496626, '1-8-1': -40.74087840802368, '1-13-1': -36.02033103243854, '7-17-2': -31.12797195665386, '15-10-2': -52.29600146778428, '7-3-2': -46.62856314166048, '4-4-1': -44.76785733034799, '15-8-0': -43.434194331431165, '13-6-1': -50.61667963511428, '13-5-2': -46.54211398109621, '12-5-1': -49.84136540824996, '11-4-0': -48.11748841664564, '15-12-2': -53.31042030862163, '12-5-2': -49.3451833920918, '3-5-2': -44.07500656678899, '15-9-2': -51.56894232759495, '14-7-1': -49.95438655356126, '2-14-1': -34.56511371622191, '8-3-2': -46.754505305961466, '6-3-1': -45.75140784009528, '3-4-1': -42.855766788747815, '2-5-2': -42.09659037130865, '1-13-2': -35.724326664349576, '3-15-1': -34.558210161405434, '2-14-2': -34.864339235054004, '12-17-0': -34.74359689933285, '6-17-2': -31.09915533110861, '12-17-2': -24.39104050333354, '13-16-2': -35.68060888433616, '15-14-0': -40.14230474858897, '16-12-0': -47.31711514463946, '16-11-1': -46.10679404566381, '16-11-2': -50.69653997413971, '11-17-1': -27.159059234191172, '13-16-1': -23.29608784941489, '14-15-2': -25.20093821766515, '15-13-2': -35.14236051063923, '16-13-1': -39.8658436271444, '16-12-2': -41.010479820356466, '16-12-1': -47.34151060640218, '16-11-0': -50.2233525959415, '16-10-0': -49.86694392193404, '16-9-0': -43.057856241915815, '14-15-1': -36.61776276832256, '14-7-2': -50.86464784798322, '1-6-1': -40.78632298499871, '9-3-2': -47.05442693110242, '9-3-0': -46.92462058864781, '2-15-0': -33.335652856909505, '3-15-2': -33.44705606634225, '3-16-1': -31.81117225954859, '5-17-0': -30.44850445762427, '9-18-1': -28.87255382526455, '12-17-1': -30.735042453597057, '14-16-0': -30.85209356188786, '16-10-2': -49.76757882327043, '16-10-1': -44.31598940142211, '16-9-2': -47.97819669590522, '10-18-0': -28.903585491358726, '15-14-2': -41.97562245525444, '15-8-2': -49.01434518273073, '11-4-1': -48.25931606224822, '7-3-1': -46.31974868584146, '2-15-1': -33.34147889922357, '5-17-1': -30.16803769087554, '15-14-1': -31.800135872732856, '16-13-0': -39.50742531443459, '16-9-1': -48.735371715653564, '15-7-0': -41.46466671681858, '14-6-2': -47.72280636944571, '14-6-0': -42.960624752123735, '13-5-0': -43.98587476310818, '11-4-2': -48.2955060480368, '3-16-0': -31.706752221469255, '2-5-1': -42.40695446982587, '2-15-2': -32.87972081646833, '3-16-2': -31.00815535637857, '8-3-1': -47.24042078237163, '1-14-1': -34.49519407529083, '8-18-0': -27.96620677310192, '9-18-0': -28.403658789034253, '4-17-0': -30.192584417391842, '8-18-2': -27.629670016287008, '15-15-0': -32.89289962844409, '5-17-2': -28.827615032359553, '7-18-0': -28.062284616946304, '8-18-1': -27.766134901906184, '9-18-2': -27.286944478867913, '14-6-1': -47.55990156421214, '12-4-0': -41.758469609441846, '11-3-2': -43.65671130290516, '10-3-0': -46.980151115066406, '5-3-2': -44.291961528726496, '5-3-1': -44.44024462066148, '1-6-2': -40.86691022803314, '13-17-1': -28.2339341394942, '13-17-0': -30.179108705047444, '14-16-1': -25.207349103741944, '16-14-0': -30.73842681085131, '17-12-1': -40.57055571317663, '17-12-2': -39.989751440868275, '17-12-0': -40.927525267246736, '17-11-0': -44.42257792895389, '17-10-1': -43.23880115825709, '17-10-0': -41.73652826102436, '17-9-0': -40.73473888267733, '16-8-2': -45.66600593424644, '16-8-0': -44.239078281014315, '15-7-2': -48.93584293848021, '8-2-0': -40.80399204434118, '7-2-2': -43.98496793019855, '6-2-0': -41.42748646689302, '5-2-2': -42.19431558177467, '4-3-0': -43.46733136968887, '7-18-1': -26.867289788764555, '4-16-2': -30.7516102581207, '10-18-1': -27.843498541621162, '16-13-2': -36.295117803515595, '17-13-1': -33.30062428826731, '17-13-0': -32.71669686009042, '17-10-2': -43.87797973090457, '17-8-0': -40.27686097153921, '16-8-1': -41.087450619157295, '16-7-0': -40.187947771475805, '16-7-2': -43.83380392548664, '15-6-0': -40.61607203277483, '14-5-0': -40.480761435154264, '13-4-2': -43.67919002773031, '13-4-1': -40.37242060660391, '12-4-2': -45.870550451418886, '11-3-0': -39.79661396992188, '10-2-0': -42.131765081834494, '9-2-2': -43.3985092976772, '7-2-1': -43.70969773062976, '4-2-0': -40.471511974827735, '15-7-1': -48.288014109095094, '14-5-2': -43.09350589661438, '10-3-2': -46.80808699165852, '4-3-1': -43.46687203809094, '2-4-1': -41.61800583702513, '1-5-2': -41.29247611175531, '15-15-2': -19.94636849264809, '12-4-1': -46.37682543229817, '11-3-1': -43.64969349732163, '10-3-1': -44.410234728106516, '13-5-1': -48.17107182005495, '7-2-0': -41.495338168049194, '6-2-2': -43.16919826020337, '17-11-1': -43.8138688720301, '16-14-2': -17.67681635710925, '4-3-2': -43.41684673322694, '4-17-2': -29.432117256014607, '6-18-0': -28.518766756846684, '10-18-2': -26.61282225382628, '11-18-1': -26.63676896806105, '11-18-0': -26.716264328458017, '14-16-2': -20.560386225919878, '15-16-0': -17.183864287946243, '17-9-2': -42.31592229930788, '13-4-0': -43.44910197889939, '12-3-0': -41.33912656523226, '17-13-2': -29.370586606439645, '18-12-0': -35.6768879467603, '18-11-0': -38.10353958032593, '18-10-0': -39.10333860781118, '17-8-2': -40.36032288565521, '9-2-0': -39.81976048924964, '6-2-1': -43.54537129775878, '2-4-0': -41.594288983206816, '16-15-0': -17.45383329593141, '16-14-1': -29.431354783995147, '17-14-2': -12.406609043419362, '18-13-0': -16.40050373972041, '18-12-1': -35.655335714322646, '18-11-2': -38.112241994916566, '18-10-2': -39.39210061802923, '18-10-1': -39.08504212147695, '18-9-0': -37.605297398408176, '1-14-0': -34.73147966537138, '17-11-2': -41.580554698243404, '6-18-2': -26.81838586235956, '7-18-2': -28.35798170140586, '12-18-0': -21.455168797253265, '11-18-2': -25.347282755877895, '16-15-1': -17.418063685269008, '17-14-0': -20.44790823382718, '16-7-1': -43.87517820324515, '16-6-0': -39.188932739047345, '8-2-2': -44.29690131195321, '14-5-1': -43.03934929983147, '15-15-1': -34.222047670254014, '18-12-2': -35.375186250285665, '18-11-1': -38.243898391197305, '17-9-1': -43.34090191183583, '15-6-1': -43.356199889626374, '19-12-0': -26.983872820616543, '19-11-0': -29.352783392975276, '19-11-1': -29.34510280901488, '19-11-2': -29.35902384124661, '19-10-0': -29.81963665052886, '19-10-2': -29.66727213741782, '19-10-1': -30.570579300704775, '18-9-1': -37.699084623476196, '18-8-1': -34.03523176870726, '17-8-1': -40.30943439413647, '8-2-1': -43.48339575409191, '5-2-1': -41.771155196441555, '5-2-0': -40.67811863859714, '4-2-2': -41.62781637533124, '3-3-0': -42.35638521260784, '2-3-0': -39.53600510349924, '15-6-2': -44.81644782875537, '12-18-2': -21.343149497046582, '13-17-2': -22.700858307543218, '2-4-2': -41.54633210818256, '1-5-1': -41.30799407203138, '18-9-2': -37.68948636706374, '18-8-0': -34.1677732275442, '17-7-1': -37.56380285416228, '18-13-1': -18.176206430730918, '1-4-0': -40.10703856654167, '1-14-2': -34.71017155486554, '6-18-1': -28.579318816875055, '12-18-1': -21.26469479876597, '15-16-1': -16.658474355273075, '4-17-1': -30.298859807272503, '14-17-0': -18.443338600652282, '14-17-2': -18.4223071764845, '18-14-0': -10.293281825978859, '18-13-2': -12.180993883400898, '19-13-0': -10.720148468746391, '19-13-1': -10.875096273285823, '19-13-2': -8.183355064253961, '13-18-0': -19.887282920774137, '1-4-2': -40.74488058227865, '16-15-2': -15.885805655046255, '17-14-1': -23.72810932953801, '9-19-0': -24.79127729210388, '17-15-0': -11.584083361737585, '9-2-1': -43.19520347344096, '17-7-0': -37.5287366950444, '15-5-0': -39.65747999114292, '5-18-0': -28.22591363253308, '9-19-2': -21.60014705346026, '10-19-0': -23.184386514424745, '18-14-1': -9.862109048827904, '18-14-2': -7.936289566069858, '19-14-2': -3.9850234724358993, '19-14-0': -6.602934250598334, '1-4-1': -40.032603396424506, '14-17-1': -18.561398695191002, '4-2-1': -41.02522770215955, '3-3-2': -42.39315081943021, '10-2-2': -42.32060956133516, '12-3-1': -40.508320587576534, '14-4-0': -39.22228056226081, '19-12-1': -27.048121925612136, '15-16-2': -15.48064735274824, '16-16-0': -13.794972342851967, '8-19-0': -25.1985270622012, '13-18-1': -17.60477369042077, '15-17-0': -13.825503858280435, '19-12-2': -26.8194644940554, '16-16-1': -13.089859660809415, '13-18-2': -19.814747742249864, '19-9-2': -17.954235193081413, '19-9-1': -18.942622349214815, '19-9-0': -18.350810083657564, '18-8-2': -33.76195515890534, '5-18-1': -27.32142719571335, '17-15-1': -10.466310033533842, '8-19-2': -22.989290089887962, '9-19-1': -24.52167355736809, '17-15-2': -9.287998538418053, '10-19-2': -20.5548519434683, '11-19-1': -21.88816815496278, '11-19-2': -19.67782498273185, '11-19-0': -21.60932683363698, '14-18-0': -17.19407195981334, '18-15-0': -7.6370250396969475, '12-19-0': -19.311496943568823, '14-18-2': -16.613530741062238, '15-17-2': -15.287569070758373, '16-17-0': -11.554790621895076, '19-14-1': -6.265684959650905, '16-17-1': -11.548120375551596, '17-16-1': -8.923480940694057, '17-16-0': -9.495248871133066, '16-16-2': -13.317552356373492, '18-15-1': -7.270075424322628, '19-15-0': -4.5584428488237645, '15-17-1': -15.236493259365158, '17-16-2': -9.312382552468465, '19-15-1': -3.976196199712224, '18-16-0': -6.1829561269026865, '18-16-1': -6.782458318968999, '14-18-1': -15.384558621410218, '18-15-2': -6.375061225910814, '19-15-2': -5.021773547714869, '18-16-2': -6.667034471610837, '15-5-2': -39.761157636047244, '13-3-0': -34.13875881025336, '11-2-0': -37.66180297668009, '10-2-1': -40.76524017291566, '9-1-0': -29.35386109777498, '8-1-0': -39.056030509552116, '7-1-0': -39.77954021224924, '11-2-1': -37.89576368143743, '7-1-2': -39.81546081761905, '6-1-0': -39.36950207500872, '5-1-0': -39.089706660983296, '3-2-0': -39.724119212557795, '2-2-0': -39.19280332434014, '1-2-0': -35.71045737295755, '1-3-0': -39.39899800552871, '12-3-2': -41.108092811581606, '2-3-2': -40.659642669980265, '17-7-2': -37.42168527474893, '2-3-1': -39.6981851444263, '3-3-1': -42.049585863974904, '7-19-0': -25.4014506282624, '1-3-2': -39.384373358068046, '3-2-1': -39.86671361615733, '4-1-0': -32.867637115039855, '3-1-0': -21.50548632421957, '5-1-2': -39.14802784674143, '16-6-2': -38.976901919130555, '5-1-1': -38.993081714013954, '2-2-2': -39.24839957323814, '8-1-2': -38.92160314863094, '3-2-2': -40.1683493072605, '19-16-0': -3.85843043125546, '5-18-2': -28.702729586178748, '1-3-1': -39.406997527014184, '6-1-2': -39.55785667904159, '13-3-2': -34.041333993844106, '11-2-2': -37.95107249180512, '6-1-1': -39.38625332526939, '16-17-2': -11.151494446738177, '8-19-1': -25.180022739261616, '19-16-1': -4.185421402449527, '10-19-1': -23.6869726779441, '7-1-1': -39.81597115101496, '17-17-0': -7.0153590551967255, '7-19-1': -24.3875653245465, '13-3-1': -34.27756424769797, '10-1-0': -5.832133195939647, '12-2-0': -9.276914313826584, '8-1-1': -38.792459387958225, '9-1-2': -30.210067226308077, '10-1-1': -2.9123480122441343, '15-18-0': -9.878854423717218, '10-1-2': -3.775104023172986, '19-16-2': -4.385616668757325, '7-19-2': -25.404038000784087, '12-19-1': -19.376926538951587, '13-19-0': -17.067720955831685, '12-19-2': -17.55800992803797, '15-5-1': -39.74807028558273, '2-2-1': -39.2266823424431, '2-1-0': -5.391810520638389, '4-1-2': -33.22057175176026, '15-18-1': -10.184765663884995, '17-17-1': -7.05874309190869, '13-19-2': -16.903420518812204, '6-19-0': -21.98553574762631, '13-19-1': -16.326323057014378, '17-17-2': -6.9206329009274645, '19-17-0': -0.9282102012308148, '18-17-0': -1.084111521557921, '14-4-2': -39.31599676723591, '18-7-0': -6.1514295738098586, '16-6-1': -39.07468640629577, '14-4-1': -39.25371139890772}

# test run:


# logger.info('###### TESTING: ######')
#
# logger.setLevel(logging.INFO)
#
# qTable2 = q_learn(env,
#                  min_alpha=0,       # temperature/learning rate
#                  alpha_decay=1,     # reduction factor per episode, was 0.003
#                  gamma=0.99,        # discount factor, was 0.99
#                  min_epsilon=0.0,   # minimal epsilon (exploration rate for e-greedy policy)
#                  epsilon_decay=1,   # reduction factor per episode, was 0.003
#                  episodes=3,
#                  max_steps=20000,   # abort episode after this number of steps
#                  q_table=qTable,
#                  average_rewards=False)


# print_q2(qTable)


# qTable = q_learn(env,
#                  min_alpha=0.01,  # temperature/learning rate
#                  gamma=0.95,  # discount factor, was 0.95
#                  min_epsilon=0.03,    # minimal epsilon (exploration rate for e-greedy policy)
#                  decay=0.995,  #  reduction factor per episode, was 0.997
#                  episodes=1000,
#                  max_steps=2000,  # abort episode after this number of steps
#                  q_table={})


logger.info(q_table)

# (year, 'human:', i, h.age, fundsBefore, h.funds, h.happiness, 'reward:', r, 'company:', companies[0].funds, companies[0].reputation)

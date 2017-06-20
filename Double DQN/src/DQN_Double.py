import tensorflow as tf
import numpy as np

# Hyperparameters
BATCH_SIZE = 32
GREEDY_POLICY_EPSILON = 0.9
DISCOUNT_FACTOR_GAMMA = 0.9
MAX_MEMORY_CAPACITY = 200
MEMORY_COUNTER = 0
LEARNING_STEP_COUNTER = 0
LEARNING_RATE = 0.01
NUMBER_OF_ACTIONS = 3
NUMBER_OF_STATES = 4
TARGET_UPDATE_INTERVAL = 500

# Initializing Memory
MEMORYS = np.zeros((MAX_MEMORY_CAPACITY, NUMBER_OF_STATES, 80, 80))
MEMORYA = np.zeros((MAX_MEMORY_CAPACITY, 1), dtype=int)
MEMORYR = np.zeros((MAX_MEMORY_CAPACITY, 1))
MEMORYS_ = np.zeros((MAX_MEMORY_CAPACITY, NUMBER_OF_STATES, 80, 80))


def make_conv(inputlayer):
    # Convolutional Layer #1
    convolutional_layer_1 = tf.layers.conv2d(
        inputs=inputlayer,  # inputs --> last layer
        strides=4,
        filters=32,  # filters --> number of output layers OR new depth
        kernel_size=[8, 8],  # kernel_size --> size of each scanning area
        padding="same",  # padding --> edge processing method
        activation=tf.nn.relu
    )

    # Pooling Layer #1
    pooling_layer_1 = tf.layers.max_pooling2d(
        inputs=convolutional_layer_1,  # inputs --> last layer
        pool_size=[4, 4],
        strides=2
    )

    # Convolutional Layer #2
    convolutional_layer_2 = tf.layers.conv2d(
        inputs=pooling_layer_1,  # inputs --> last layer
        strides=2,
        filters=32,  # filters --> number of output layers OR new depth
        kernel_size=[4, 4],  # kernel_size --> size of each scanning area
        padding="same",  # padding --> edge processing method
        activation=tf.nn.relu
    )

    # Pooling Layer #2
    pooling_layer_2 = tf.layers.max_pooling2d(
        inputs=convolutional_layer_2,  # inputs --> last layer
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional Layer #3
    convolutional_layer_3 = tf.layers.conv2d(
        inputs=pooling_layer_2,  # inputs --> last layer
        strides=1,
        filters=64,  # filters --> number of output layers OR new depth
        kernel_size=[3, 3],  # kernel_size --> size of each scanning area
        padding="same",  # padding --> edge processing method
        activation=tf.nn.relu
    )

    # Pooling Layer #3
    pooling_layer_3 = tf.layers.max_pooling2d(
        inputs=convolutional_layer_3,  # inputs --> last layer
        pool_size=[2, 2],
        strides=1
    )

    pooling_layer_3_flat = tf.reshape(pooling_layer_3, [-1, 64])
    return pooling_layer_3_flat


class DeepQNetwork:
    """
    This initialize the neural networks
    state --> output frame set from the game engine [Tensor of Frame Stack]
    mode --> start training control variable [True / False]

    """

    def __init__(self):
        # tf placeholders
        self.current_state = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_STATES, 80, 80], name='CUR_STATE')
        self.input_layer = tf.transpose(self.current_state, perm=[0, 2, 3, 1], name='INPUT_LAYER')
        self.action = tf.placeholder(tf.int32, shape=[None, 1], name='ACTIONS')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='REWARD')
        self.next_state = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_STATES, 80, 80], name='NEXT_STATE')
        self.input_layer_ = tf.transpose(self.next_state, perm=[0, 2, 3, 1])

        """
        Step 1
        Processing input frames using CNN
        Number of convolutional layers = 2
        
        """
        evalconv = make_conv(self.input_layer)
        targetconv = make_conv(self.input_layer_)

        """
        Step 2
        Create a dense layer [Target + Evaluation]
        It is a replacement of Q-Table in Q-Learning by a neural network
        And therefore called Deep Q-Network
        
        """

        # Target Dense Layer
        with tf.variable_scope('q'):  # current network
            self.evaluation_layer = tf.layers.dense(
                inputs=evalconv,
                units=1024,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0, 0.1)
            )

            self.evaluated_q_value = tf.layers.dense(
                inputs=self.evaluation_layer,
                units=NUMBER_OF_ACTIONS,
                kernel_initializer=tf.random_normal_initializer(0, 0.1)
            )

        with tf.variable_scope('q_next'):  # older network, not to train
            self.target_layer = tf.layers.dense(
                inputs=targetconv,
                units=1024,
                activation=tf.nn.relu,
                trainable=False
            )

            self.q_next = tf.layers.dense(
                inputs=self.target_layer,
                units=NUMBER_OF_ACTIONS,
                trainable=False
            )

        q_target = self.reward + DISCOUNT_FACTOR_GAMMA * tf.reduce_max(self.q_next, axis=1)  # shape=(None, ),
        a_one_hot = tf.squeeze(tf.one_hot(self.action, depth=NUMBER_OF_ACTIONS, dtype=tf.float32))
        q_wrt_a = tf.reduce_sum(tf.multiply(self.evaluated_q_value, a_one_hot),
                                axis=1)  # shape=(None, ), q for current state

        loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))
        self.train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __choose_action(self, state):
        # current_state = state[np.newaxis, :]
        if np.random.uniform() < GREEDY_POLICY_EPSILON:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.evaluated_q_value, feed_dict={self.current_state: [state]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, NUMBER_OF_ACTIONS)
        return action

    @staticmethod
    def __store_transition(s, a, r, s_):
        global MEMORY_COUNTER
        # replace the old memory with new memory
        index = MEMORY_COUNTER % MAX_MEMORY_CAPACITY
        MEMORYS[index] = s
        MEMORYA[index] = a
        MEMORYR[index] = r
        MEMORYS_[index] = s_
        MEMORY_COUNTER += 1

    def __learn(self):
        # update target net
        global LEARNING_STEP_COUNTER
        if LEARNING_STEP_COUNTER % TARGET_UPDATE_INTERVAL == 0:
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        LEARNING_STEP_COUNTER += 1

        # learning
        sample_index = np.random.choice(MAX_MEMORY_CAPACITY, BATCH_SIZE)
        b_s = MEMORYS[sample_index]
        b_a = MEMORYA[sample_index].astype(int)
        b_r = MEMORYR[sample_index]
        b_s_ = MEMORYS_[sample_index]
        self.sess.run(
            self.train_op,
            {
                self.current_state: b_s,
                self.action: b_a,
                self.reward: b_r,
                self.next_state: b_s_
            }
        )

    def start_playing(self, env):
        print('\nCollecting experience...')
        i_episode = 0
        while True:
            curr_state = env.restart()
            episode_reward = 0
            while True:
                action = self.__choose_action(curr_state)
                next_state = np.zeros((4, 80, 80), dtype=float)
                rewards = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
                # take action
                next_state[0], rewards[0], is_game_over = env.render(self.action)
                if not is_game_over:
                    next_state[1], rewards[1], is_game_over = env.render(self.action)
                else:
                    next_state[1], rewards[1] = next_state[0], 0
                if not is_game_over:
                    next_state[2], rewards[2], is_game_over = env.render(self.action)
                else:
                    next_state[2], rewards[2] = next_state[1], 0
                if not is_game_over:
                    next_state[3], rewards[3], is_game_over = env.render(self.action)
                else:
                    next_state[3], rewards[3] = next_state[2], 0
                reward = np.sum(rewards)

                """
                # modify the reward
                x, x_dot, theta, theta_dot = next_state
                reward_1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
                reward_2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
                reward = reward_1 + reward_2
                """

                self.__store_transition(curr_state, action, reward, next_state)

                episode_reward += reward
                if MEMORY_COUNTER > MAX_MEMORY_CAPACITY:
                    self.__learn()
                    if is_game_over:
                        print('Episode: ', i_episode, ' | Episode Rewards: ', round(episode_reward, 2))

                if is_game_over:
                    break
                curr_state = next_state

                i_episode += 1

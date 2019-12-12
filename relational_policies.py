from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy, mlp_extractor
from stable_baselines.a2c.utils import linear
import tensorflow as tf
from utils import get_coor, MHDPA, residual_block, rrl_cnn, boxworld_cnn, simple_cnn, reduce_border_extractor, MHDPAv2, residual_blockv2
from stable_baselines.a2c.utils import batch_to_seq, seq_to_batch, lstm


class RelationalPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=boxworld_cnn, feature_extraction="cnn", **kwargs):
        super(RelationalPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                               scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)
        with tf.variable_scope("model", reuse=reuse):
            print('self.processed_obs', self.processed_obs)
            # [B,H,W,Deepth]
            extracted_features = cnn_extractor(self.processed_obs, **kwargs)
            print('extracted_features', extracted_features)
            relation_block_output = self.relation_block(extracted_features)
            # original code
            net_arch = [128, dict(vf=[256], pi=[16])]
            # pi_latent = vf_latent = cnn_extractor(residual_maxpooling_output, **kwargs)
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(relation_block_output), net_arch, act_fun)
            self._value_fn = linear(vf_latent, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    def attention(self, obs, state=None, mask=None):
        return self.sess.run(self.weights, {self.obs_ph: obs})


class RelationalLstmPolicy(RecurrentActorCriticPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, cnn_extractor=simple_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(RelationalLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                   state_shape=(2 * n_lstm, ), reuse=reuse,
                                                   scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            print('self.processed_obs', self.processed_obs)
            # [B,H,W,Deepth]
            # extracted_features = cnn_extractor(self.processed_obs, **kwargs)
            # relation_block_output = self.relation_block(extracted_features)
            # test reduce_relation_block
            relation_block_output = self.reduce_relation_block(self.processed_obs)

            # original code
            input_sequence = batch_to_seq(relation_block_output, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            value_fn = linear(rnn_output, 'vf', 1)
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

        self._value_fn = value_fn

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def attention(self, obs, state=None, mask=None):
        return self.sess.run(self.weights, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


def relation_block(self, extracted_features):
    coor = get_coor(extracted_features)
    # [B,Height,W,D+2]
    entities = tf.concat([extracted_features, coor], axis=3)
    print('entities:', entities)
    # [B,H*W,num_heads,Deepth=D+2]
    MHDPA_output, weights = MHDPA(entities, "MHDPA", num_heads=2)
    print('MHDPA_output', MHDPA_output)
    self.weights = weights
    # [B,H*W,num_heads,Deepth]
    residual_output = residual_block(entities, MHDPA_output)
    print('residual_output', residual_output)

    # max_pooling
    residual_maxpooling_output = tf.reduce_max(residual_output, axis=[1])
    print('residual_maxpooling_output', residual_maxpooling_output)
    return residual_maxpooling_output


def reduce_relation_block(self, processed_obs):
    coor = get_coor(processed_obs)
    # [B,Height,W,D+2]
    processed_obs = tf.concat([processed_obs, coor], axis=3)
    # [B,N,W,D+2 N=Height*w+1
    entities = reduce_border_extractor(processed_obs)
    # [B,N,num_heads,Deepth=D+2]
    MHDPA_output, weights = MHDPAv2(entities, "MHDPA", num_heads=2)
    print('MHDPA_output', MHDPA_output)
    self.weights = weights
    # [B,N,num_heads,Deepth]
    residual_output = residual_blockv2(entities, MHDPA_output)
    print('residual_output', residual_output)

    # # max_pooling
    # residual_maxpooling_output = tf.reduce_max(residual_output, axis=[1])
    # print('residual_maxpooling_output', residual_maxpooling_output)
    # return residual_maxpooling_output
    # average_pooling
    residual_avepooling_output = tf.reduce_mean(residual_output, axis=[1])
    print('residual_avepooling_output', residual_avepooling_output)
    return residual_avepooling_output


ActorCriticPolicy.relation_block = relation_block
ActorCriticPolicy.reduce_relation_block = reduce_relation_block

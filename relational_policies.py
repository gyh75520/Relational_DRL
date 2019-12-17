from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import linear
import tensorflow as tf
from utils import MHDPA, residual_block, build_entities, layerNorm
from stable_baselines.a2c.utils import batch_to_seq, seq_to_batch, lstm


class RelationalPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, net_arch=None,
                 act_fun=tf.tanh, feature_extraction="cnn", **kwargs):
        super(RelationalPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                               scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)
        with tf.variable_scope("model", reuse=reuse):
            print('self.processed_obs', self.processed_obs)
            relation_block_output = self.relation_block(self.processed_obs)
            pi_latent = vf_latent = tf.layers.flatten(relation_block_output)
            # original code
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
        return self.sess.run(self.relations, {self.obs_ph: obs})


class RelationalLstmPolicy(RecurrentActorCriticPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(RelationalLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                   state_shape=(2 * n_lstm, ), reuse=reuse,
                                                   scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        with tf.variable_scope("model", reuse=reuse):
            print('self.processed_obs', self.processed_obs)
            relation_block_output = self.relation_block(self.processed_obs)

            # original code
            input_sequence = batch_to_seq(relation_block_output, self.n_env, n_steps)
            print('input_sequence', input_sequence)
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
        return self.sess.run(self.relations, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


def relation_block(self, processed_obs):
    entities = build_entities(processed_obs, self.reduce_obs)
    print('entities:', entities)
    # [B,n_heads,N,Deepth=D+2]
    MHDPA_output, self.relations = MHDPA(entities, n_heads=2)
    print('MHDPA_output', MHDPA_output)
    # [B,n_heads,N,Deepth]
    residual_output = residual_block(entities, MHDPA_output)
    print('residual_output', residual_output)
    # max_pooling [B,n_heads,N,Deepth] --> [B,n_heads,Deepth]
    maxpooling_output = tf.reduce_max(residual_output, axis=2)
    print('maxpooling_output', maxpooling_output)
    # [B,n_heads*Deepth]
    # output = tf.layers.flatten(maxpooling_output)
    # output = layerNorm(output, "relation_layerNorm")
    # print('relation_layerNorm', output)
    return maxpooling_output

    # # average_pooling
    # maxpooling_output = tf.reduce_mean(residual_output, axis=2)
    # print('maxpooling_output', maxpooling_output)
    # return maxpooling_output


ActorCriticPolicy.relation_block = relation_block

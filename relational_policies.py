from stable_baselines.common.policies import ActorCriticPolicy, mlp_extractor
from stable_baselines.a2c.utils import linear
import tensorflow as tf
from utils import get_coor, MHDPA, residual_block, nature_cnn


class RelationalPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, net_arch=None,
                 act_fun=tf.nn.relu, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(RelationalPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                               scale=(feature_extraction == "cnn"))
        self._kwargs_check(feature_extraction, kwargs)
        with tf.variable_scope("model", reuse=reuse):
            print('self.processed_obs', self.processed_obs)
            # [B,H,W,Deepth]
            extracted_features = cnn_extractor(self.processed_obs, **kwargs)
            print('extracted_features', extracted_features)
            coor = get_coor(extracted_features)
            # [B,Height,W,D+2]
            entities = tf.concat([extracted_features, coor], axis=3)
            print('entities:', entities)
            # [B,H*W,num_heads,Deepth=D+2]
            MHDPA_output, weights = MHDPA(entities, "extracted_features", num_heads=2)
            print('MHDPA_output', MHDPA_output)
            self.attention = weights
            # [B,H*W,num_heads,Deepth]
            residual_output = residual_block(entities, MHDPA_output)
            print('residual_output', residual_output)

            # max_pooling
            residual_maxpooling_output = tf.reduce_max(residual_output, axis=[1])
            print('residual_maxpooling_output', residual_maxpooling_output)

            # if net_arch is None:
            #     layers = [64, 64]
            # net_arch = [dict(vf=layers, pi=layers)]
            net_arch = [128, dict(vf=[256], pi=[16])]

            # pi_latent = vf_latent = cnn_extractor(residual_maxpooling_output, **kwargs)
            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(residual_maxpooling_output), net_arch, act_fun)

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

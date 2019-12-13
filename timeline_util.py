import tensorflow as tf
from tensorflow.python.client import timeline


def _train_step(self, obs, states, rewards, masks, actions, values, update, writer=None):
    '''
    add timeline(performance analysis) but  make tensorboard_log expire
    '''
    advs = rewards - values
    cur_lr = None
    for _ in range(len(obs)):
        cur_lr = self.learning_rate_schedule.value()
    assert cur_lr is not None, "Error: the observation input array cannon be empty"

    td_map = {self.train_model.obs_ph: obs, self.actions_ph: actions, self.advs_ph: advs,
              self.rewards_ph: rewards, self.learning_rate_ph: cur_lr}
    if states is not None:
        td_map[self.train_model.states_ph] = states
        td_map[self.train_model.dones_ph] = masks
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # run loss backprop with timeline, but once every 100 runs save the metadata
    log_interval = 100  # same with verbose
    if update % log_interval == 0 or update == 1:
        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map, options=run_options, run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(self.log_dir + 'timeline_step_%d.json' % (update * self.n_batch), 'w') as f:
            f.write(chrome_trace)
    else:
        policy_loss, value_loss, policy_entropy, _ = self.sess.run(
            [self.pg_loss, self.vf_loss, self.entropy, self.apply_backprop], td_map)

    return policy_loss, value_loss, policy_entropy

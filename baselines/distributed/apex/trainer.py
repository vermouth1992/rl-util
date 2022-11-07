import ray
from ray.util import queue

from baselines.distributed.apex.actor import Actor
from baselines.distributed.apex.learner import Learner
from baselines.distributed.apex.logger_tester import Logger
from baselines.distributed.apex.replay_manager import ReplayManager


def run_apex(make_actor_fn_lst,  # each actor may be different (e.g. exploration)
             make_learner_fn,
             make_test_agent_fn,
             make_sampler_fn,
             make_local_buffer_fn,
             make_replay_fn,
             make_tester_fn,
             set_thread_fn,
             set_weights_fn,
             get_weights_fn,
             exp_name,
             config=None,
             # actor args
             num_cpus_per_actor=1,
             weight_update_freq=20,
             # learner args
             num_cpus_per_learner=1,
             num_gpus_per_learner=1,
             batch_size=256,
             weight_push_freq=10,
             update_after=10000,
             # logging
             total_num_policy_updates=1000000,
             logging_freq=10000,
             num_test_episodes=30,
             num_cpus_tester=1,
             logger_path: str = None,
             seed=1):
    if config is None:
        config = locals()

    Actor_remote = ray.remote(num_cpus=num_cpus_per_actor)(Actor)
    Learner_remote = ray.remote(num_cpus=num_cpus_per_learner, num_gpus=num_gpus_per_learner)(Learner)
    ReplayManager_remote = ray.remote(num_cpus=1)(ReplayManager)

    # create replay manager
    replay_manager = ReplayManager_remote.remote(make_replay_fn=make_replay_fn,
                                                 update_after=update_after)

    # make test queue
    testing_queue = queue.Queue(maxsize=3)

    # create learners
    learner = Learner_remote.remote(
        make_agent_fn=make_learner_fn,
        replay_manager=replay_manager,
        set_thread_fn=set_thread_fn,
        get_weights_fn=get_weights_fn,
        testing_queue=testing_queue,
        logging_freq=logging_freq,
        weight_push_freq=weight_push_freq,
        batch_size=batch_size,
        prefetch=10,
        prefetch_rate_limit=0.99,
        num_threads=num_cpus_per_learner
    )

    # create actors
    actors = [Actor_remote.remote(
        make_agent_fn=make_actor_fn,
        make_sampler_fn=make_sampler_fn,
        make_local_buffer_fn=make_local_buffer_fn,
        weight_server_lst=[learner],
        replay_manager_lst=[replay_manager],
        set_thread_fn=set_thread_fn,
        set_weights_fn=set_weights_fn,
        weight_update_freq=weight_update_freq,
        num_threads=num_cpus_per_actor,
    ) for make_actor_fn in make_actor_fn_lst]

    # create logger
    set_thread_fn(num_cpus_tester)
    logger = Logger(
        receive_queue=testing_queue,
        set_weights_fn=set_weights_fn,
        set_thread_fn=set_thread_fn,
        logging_freq=logging_freq,
        total_num_policy_updates=total_num_policy_updates,
        actors=actors,
        local_learners=[learner],
        replay_manager_lst=[replay_manager],
        num_cpus_tester=num_cpus_tester,
        make_test_agent_fn=make_test_agent_fn,
        make_tester_fn=make_tester_fn,
        num_test_episodes=num_test_episodes,
        exp_name=exp_name,
        seed=seed,
        logger_path=logger_path,
        config=config
    )

    # start to run actors
    for actor in actors:
        actor.run.remote()

    learner.run.remote()
    logger.log()

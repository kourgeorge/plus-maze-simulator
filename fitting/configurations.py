import config

def initial_config():
    return {
        'REPORTING_INTERVAL': config.REPORTING_INTERVAL,
        'LEARNING_RATE': config.LEARNING_RATE,
        'EXPLORATION_EPSILON': config.EXPLORATION_EPSILON,
        'BATCH_SIZE': config.BATCH_SIZE,
        'SUCCESS_CRITERION_THRESHOLD': config.SUCCESS_CRITERION_THRESHOLD,
        'MEMORY_SIZE': config.MEMORY_SIZE,
        'STIMULIENCODINGSIZE': config.STIMULIENCODINGSIZE,
        'FORGETTING': config.FORGETTING,
        'MOTIVATED_REWARD': config.MOTIVATED_REWARD,
        'NON_MOTIVATED_REWARD': config.NON_MOTIVATED_REWARD,
    }


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


current_config = dotdict(initial_config())


def create_dir_name(config):
    return '_'.join(map(str, [
        config.FORGETTING,
        config.NON_MOTIVATED_REWARD,
        config.LEARNING_RATE
    ]))


# def gen_get_config():
#     for MEMORY_SIZEi in range(len(MEMORY_SIZE)):
#         for FORGETTINGi in range(len(FORGETTING)):
#             for MOTIVATED_REWARDi in range(len(MOTIVATED_REWARD)):
#                 for NON_MOTIVATED_REWARDi in range(len(NON_MOTIVATED_REWARD)):
#                     for LEARNING_RATEi in range(len(LEARNING_RATE)):
#                         current_config = dotdict({
#                             'REPORTING_INTERVAL': REPORTING_INTERVAL,
#                             'BATCH_SIZE': BATCH_SIZE,
#                             'LEARNING_RATE': LEARNING_RATE[LEARNING_RATEi],
#                             'MEMORY_SIZE': MEMORY_SIZE[MEMORY_SIZEi],
#                             'STIMULIENCODINGSIZE': STIMULIENCODINGSIZE,
#                             'FORGETTING': FORGETTING[FORGETTINGi],
#                             'MOTIVATED_REWARD': MOTIVATED_REWARD[MOTIVATED_REWARDi],
#                             'NON_MOTIVATED_REWARD': NON_MOTIVATED_REWARD[NON_MOTIVATED_REWARDi],
#                             'CueType': CueType,
#                             'RewardType': RewardType
#                         })
#                         dirname = create_dir_name(current_config)
#                         yield current_config, dirname


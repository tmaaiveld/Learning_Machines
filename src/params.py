params = {
        'hardware': True,
        'load_model': True,
        'save_model': False,
        'save_data': True,

        'sens_names': ["IR" + str(i + 1) for i in range(8)],
        'ep_count': 1000,
        'step_count': 60,
        'step_size_ms': 550,
        'C': 1.0,
        'min_value': -1.,
        'max_value': 1.,
        'min_strategy': -1.,
        'max_strategy': 1.,
        'mut_prob_0': 0.92,
        'mut_prob_base': 0.3,
        'm_max': 30,
        'recovery_time': 5,
        'init_bias': True,
        'reeval_rate': 0.2,
        'max_sens': 6.4
}
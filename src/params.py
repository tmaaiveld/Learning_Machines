params = {
        'hardware': False,
        'load_model': False,
        'save_model': True,
        'save_data': True,

        'sens_names': ["IR" + str(i + 1) for i in range(8)],
        'ep_count': 1000,
        'step_count': 200,
        'step_size_ms': 100,
        'C': 2.0,
        'min_value': -1.,
        'max_value': 1.,
        'min_strategy': -1.,
        'max_strategy': 1.,
        'mut_prob_0': 0.92,
        'mut_prob_base': 0.3,
        'm_max': 20,
        'recovery_time': 5,
        'init_bias': True,
        'reeval_rate': 0.2,
        'max_sens': 6.3
}
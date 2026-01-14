from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from tensorflow.keras.optimizers.schedules import ExponentialDecay, CosineDecay

def get_optimizer(config):
    """
    获取优化器
    
    参数:
    - config: 优化器配置字典
    
    返回:
    - optimizer: 优化器实例
    """
    optimizer_type = config.get('type', 'adam')
    learning_rate = config.get('learning_rate', 0.001)
    decay_type = config.get('decay_type', None)
    decay_params = config.get('decay_params', {})
    
    # 学习率衰减
    if decay_type == 'exponential':
        lr_schedule = ExponentialDecay(
            initial_learning_rate=learning_rate,
            **decay_params
        )
        learning_rate = lr_schedule
    elif decay_type == 'cosine':
        lr_schedule = CosineDecay(
            initial_learning_rate=learning_rate,
            **decay_params
        )
        learning_rate = lr_schedule
    
    # 根据优化器类型创建优化器
    if optimizer_type == 'adam':
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=config.get('beta_1', 0.9),
            beta_2=config.get('beta_2', 0.999),
            epsilon=config.get('epsilon', 1e-7)
        )
    elif optimizer_type == 'rmsprop':
        optimizer = RMSprop(
            learning_rate=learning_rate,
            rho=config.get('rho', 0.9),
            momentum=config.get('momentum', 0.0),
            epsilon=config.get('epsilon', 1e-7)
        )
    elif optimizer_type == 'sgd':
        optimizer = SGD(
            learning_rate=learning_rate,
            momentum=config.get('momentum', 0.0),
            nesterov=config.get('nesterov', False)
        )
    elif optimizer_type == 'nadam':
        optimizer = Nadam(
            learning_rate=learning_rate,
            beta_1=config.get('beta_1', 0.9),
            beta_2=config.get('beta_2', 0.999),
            epsilon=config.get('epsilon', 1e-7)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer

def get_default_optimizer_config():
    """
    获取默认优化器配置
    
    返回:
    - config: 默认优化器配置字典
    """
    return {
        'type': 'adam',
        'learning_rate': 0.001,
        'decay_type': 'exponential',
        'decay_params': {
            'decay_steps': 100000,
            'decay_rate': 0.96
        },
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-7
    }

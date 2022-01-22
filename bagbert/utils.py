import tensorflow as tf

def init_env():
    global REPLICAS
    tf.get_logger().setLevel('ERROR')
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    else:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    finally:
        REPLICAS = strategy.num_replicas_in_sync
        print(f"{'='*80}\nREPLICAS: {REPLICAS}\n{'='*80}")
    return strategy
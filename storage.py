import redis


def get_database():
    return redis.StrictRedis(host='localhost', port=6379, db=0)


def save_processed(dataset_id, json):
    database = get_database()
    database.set('dot:dt:' + dataset_id, json)


def get_processed(dataset_id):
    database = get_database()
    return database.get('dot:dt:' + dataset_id)


def save_nn(dataset_id, json):
    database = get_database()
    database.set('dot:nn:' + dataset_id, json)


def get_nn(dataset_id):
    database = get_database()
    return database.get('dot:nn:' + dataset_id)


def save_training_logs(dataset_id, json):
    database = get_database()
    database.set('dot:tl:' + dataset_id, json)


def get_training_logs(dataset_id):
    database = get_database()
    return database.get('dot:tl:' + dataset_id)

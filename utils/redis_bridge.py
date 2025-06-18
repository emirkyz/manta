


import redis
import json
redis_conn = redis.Redis(host='redis', port=6379, db=0)




def update_progress_emit(progress, message, status, data_name, tid, task_obj=None,durum = None) -> bool:
    """
    Update the progress of a task and emit the status to Redis Pub/Sub.
    Args:
        progress (int): The progress percentage of the task.
        message (str): A message describing the current status of the task.
        status (str): The current status of the task.
        data_name (str): The name of the data being processed.
        tid (str): The ID of the task.
        task_obj: The task object to check for abortion status.
        durum (str, optional): Optional parameter to check for abortion status.
    Returns:
        bool: True if the task is aborted, False otherwise.
    """
    aborted = False
    if task_obj:
        if task_obj.is_aborted() or durum == "abort":
            # If the task is aborted, pass state as 'aborted'
            status = 'ABORTED'
            data = {
                "state": status,
                "task_id": tid,
                "progress": 100,
                "message": message,
                "data_name": data_name,
            }
            aborted = True
            redis_conn.publish('task_progress', json.dumps(data))
            return aborted

    # Use Redis Pub/Sub instead of HTTP request
    data = {
        "state": status,
        "task_id": tid,
        "progress": progress,
        "message": message,
        "data_name": data_name,
    }
    # Publish the progress update to a Redis channel
    redis_conn.publish('task_progress', json.dumps(data))
    return aborted

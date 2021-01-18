from ._metrics import lwlrap
from .metrics import LWLRAPCallback
from .scheduler import SchedulerCallback


__CALLBACKS__ = {
    "LWLRAPCallback": LWLRAPCallback,
    "SchedulerCallback": SchedulerCallback
}


def get_callbacks(config: dict):
    required_callbacks = config.get("callbacks")
    if required_callbacks is None:
        return []
    callbacks = []
    for callback_conf in required_callbacks:
        name = callback_conf["name"]
        params = callback_conf["params"]
        if params is None:
            params = {}
        callback_cls = __CALLBACKS__.get(name)

        if callback_cls is not None:
            callbacks.append(callback_cls(**params))
    return callbacks

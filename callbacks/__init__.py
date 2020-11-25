from .metrics import LWLRAPCallback


__CALLBACKS__ = {
    "LWLRAPCallback": LWLRAPCallback
}


def get_callbacks(config: dict):
    required_callbacks = config.get("callbacks")
    if required_callbacks is None:
        return []
    callbacks = []
    for callback_conf in required_callbacks:
        name = callback_conf["name"]
        params = callback_conf["params"]
        callback_cls = __CALLBACKS__.get(name)

        if callback_cls is not None:
            callbacks.append(callback_cls(**params))
    return callbacks

from catalyst.core import Callback, CallbackOrder, IRunner


class SchedulerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Scheduler)

    def on_loader_end(self, state: IRunner):
        state.scheduler.step()
        state.epoch_metrics["lr"] = state.scheduler.get_last_lr()

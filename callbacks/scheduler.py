from catalyst.core import Callback, CallbackOrder, IRunner


class SchedulerCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.Scheduler)

    def on_loader_end(self, state: IRunner):
        lr = state.scheduler.get_last_lr()
        import pdb
        pdb.set_trace()
        state.scheduler.step()
        print(f"LR changed from {lr}")

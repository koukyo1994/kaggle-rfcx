from catalyst.dl import Runner


class SAMRunner(Runner):

    def predict_batch(self, batch, **kwargs):
        return super().predict_batch(batch, **kwargs)

    def _handle_batch(self, batch):
        input_ = batch["image"]
        target = batch["targets"]

        input_ = input_.to(self.device)
        target["weak"] = target["weak"].to(self.device)
        target["strong"] = target["strong"].to(self.device)

        out = self.model(input_)

        loss = self.criterion(out, target)
        self.batch_metrics.update({
            "loss": loss
        })

        self.input = batch
        self.output = {"logits": out}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            self.criterion(self.model(input_), target).backward()
            self.optimizer.second_step(zero_grad=True)

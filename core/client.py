import torch
import torch.nn.functional as F
import torch.optim as optim


class Client:

    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config

    def local_train(self, global_model=None):

        device = self.config.DEVICE
        self.model.train()

        if global_model is not None:
            global_model.eval()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.LR
        )

        for _ in range(self.config.LOCAL_EPOCHS):

            for x, y in self.train_loader:

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = self.model(x)

                # ===== FedAvg mode =====
                if global_model is None:
                    loss = F.cross_entropy(logits, y)

                else:
                    with torch.no_grad():
                        teacher_logits = global_model(x)
                        teacher_probs = F.softmax(
                            teacher_logits / self.config.TEMPERATURE,
                            dim=1
                        )

                    probs = F.softmax(logits, dim=1)
                    conf, _ = torch.max(probs, dim=1)

                    # ===== Confidence mask =====
                    mask = conf > self.config.CONF_THRESHOLD

                    y_onehot = F.one_hot(y, num_classes=10).float()

                    # clean samples → hard CE
                    if mask.sum() > 0:
                        hard_loss = F.cross_entropy(
                            logits[mask],
                            y[mask]
                        )
                    else:
                        hard_loss = torch.tensor(0.0, device=device)

                    # suspicious samples → soft correction
                    if (~mask).sum() > 0:
                        soft_labels = (
                            self.config.LAMBDA * y_onehot[~mask]
                            + (1 - self.config.LAMBDA) * teacher_probs[~mask]
                        )

                        ce_soft = -(soft_labels *
                                    F.log_softmax(logits[~mask], dim=1)).sum(dim=1)

                        soft_loss = ce_soft.mean()
                    else:
                        soft_loss = torch.tensor(0.0, device=device)

                    sup_loss = hard_loss + soft_loss

                    # ===== Local KD =====
                    kd_loss = F.kl_div(
                        F.log_softmax(logits / self.config.TEMPERATURE, dim=1),
                        teacher_probs,
                        reduction="batchmean"
                    ) * (self.config.TEMPERATURE ** 2)

                    loss = sup_loss + self.config.BETA * kd_loss

                loss.backward()
                optimizer.step()

        return self.model.state_dict()

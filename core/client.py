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

                # ======================================================
                # 1️⃣ FedAvg (no global model)
                # ======================================================
                if global_model is None:
                    loss = F.cross_entropy(logits, y)

                else:

                    # ==================================================
                    # 2️⃣ FedProx
                    # ==================================================
                    if getattr(self.config, "USE_FEDPROX", False) and not self.config.USE_R2D2:

                        ce_loss = F.cross_entropy(logits, y)

                        prox_loss = 0.0
                        for w, w_global in zip(
                            self.model.parameters(),
                            global_model.parameters()
                        ):
                            prox_loss += ((w - w_global.detach()) ** 2).sum()

                        loss = ce_loss + 0.5 * self.config.MU * prox_loss

                    # ==================================================
                    # 3️⃣ R2D2
                    # ==================================================
                    elif self.config.USE_R2D2:

                        with torch.no_grad():
                            teacher_logits = global_model(x)
                            teacher_probs = F.softmax(
                                teacher_logits / self.config.TEMPERATURE,
                                dim=1
                            )

                        probs = F.softmax(logits, dim=1)
                        conf, _ = torch.max(probs, dim=1)

                        mask = conf > self.config.CONF_THRESHOLD
                        y_onehot = F.one_hot(y, num_classes=self.config.NUM_CLASSES).float()
                        # ---------- Hard samples ----------
                        if mask.sum() > 0:
                            hard_loss = F.cross_entropy(
                                logits[mask],
                                y[mask]
                            )
                        else:
                            hard_loss = torch.tensor(0.0, device=device)

                        # ---------- Suspicious samples ----------
                        if (~mask).sum() > 0:

                            if getattr(self.config, "USE_SOFT_CORRECTION", True):

                                soft_labels = (
                                    self.config.LAMBDA * y_onehot[~mask]
                                    + (1 - self.config.LAMBDA) * teacher_probs[~mask]
                                )

                                ce_soft = -(soft_labels *
                                            F.log_softmax(logits[~mask], dim=1)).sum(dim=1)

                                soft_loss = ce_soft.mean()

                            else:
                                soft_loss = F.cross_entropy(
                                    logits[~mask],
                                    y[~mask]
                                )

                        else:
                            soft_loss = torch.tensor(0.0, device=device)

                        sup_loss = hard_loss + soft_loss

                        # ---------- Local KD ----------
                        if getattr(self.config, "USE_LOCAL_KD", True):

                            kd_loss = F.kl_div(
                                F.log_softmax(
                                    logits / self.config.TEMPERATURE,
                                    dim=1
                                ),
                                teacher_probs,
                                reduction="batchmean"
                            ) * (self.config.TEMPERATURE ** 2)

                            loss = sup_loss + self.config.BETA * kd_loss

                        else:
                            loss = sup_loss

                    # ==================================================
                    # 4️⃣ Plain FedAvg fallback
                    # ==================================================
                    else:
                        loss = F.cross_entropy(logits, y)

                loss.backward()
                optimizer.step()

        return self.model.state_dict()

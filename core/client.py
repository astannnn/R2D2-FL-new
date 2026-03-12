import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


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

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LR
        )

        for _ in range(self.config.LOCAL_EPOCHS):

            for x, y in self.train_loader:

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = self.model(x)

                # =====================================================
                # 1. FedAvg (no global model)
                # =====================================================
                if global_model is None:
                    loss = F.cross_entropy(logits, y)

                else:

                    # =====================================================
                    # 2. FedProx
                    # =====================================================
                    if getattr(self.config, "USE_FEDPROX", False) and not self.config.USE_R2D2:
                        ce_loss = F.cross_entropy(logits, y)

                        prox_loss = 0.0
                        for w, w_global in zip(
                            self.model.parameters(),
                            global_model.parameters()
                        ):
                            prox_loss += ((w - w_global.detach()) ** 2).sum()

                        loss = ce_loss + 0.5 * self.config.MU * prox_loss

                    # =====================================================
                    # 3. R2D2
                    # =====================================================
                    elif getattr(self.config, "USE_R2D2", False):

                        with torch.no_grad():
                            teacher_logits = global_model(x).detach()
                            teacher_probs = F.softmax(
                                teacher_logits / self.config.TEMPERATURE,
                                dim=1
                            )

                        probs = F.softmax(logits, dim=1)
                        conf, _ = torch.max(probs, dim=1)

                        mask = conf > self.config.CONF_THRESHOLD

                        y_onehot = F.one_hot(
                            y,
                            num_classes=self.config.NUM_CLASSES
                        ).float()

                        hard_loss = torch.zeros(1, device=device)
                        soft_loss = torch.zeros(1, device=device)

                        hard_count = mask.sum()
                        soft_count = (~mask).sum()

                        if hard_count > 0:
                            hard_loss = F.cross_entropy(
                                logits[mask],
                                y[mask],
                                reduction="sum"
                            )

                        if soft_count > 0:
                            if getattr(self.config, "USE_SOFT_CORRECTION", True):
                                soft_labels = (
                                    self.config.LAMBDA * y_onehot[~mask]
                                    + (1 - self.config.LAMBDA) * teacher_probs[~mask]
                                )

                                ce_soft = -(soft_labels *
                                            F.log_softmax(logits[~mask], dim=1)).sum(dim=1)

                                soft_loss = ce_soft.sum()
                            else:
                                soft_loss = F.cross_entropy(
                                    logits[~mask],
                                    y[~mask],
                                    reduction="sum"
                                )

                        total = hard_count + soft_count

                        if total > 0:
                            sup_loss = (hard_loss + soft_loss) / total
                        else:
                            sup_loss = torch.zeros(1, device=device)

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

                    # =====================================================
                    # 4. Plain FedAvg fallback
                    # =====================================================
                    else:
                        loss = F.cross_entropy(logits, y)

                loss.backward()
                optimizer.step()

        return self.model.state_dict()

    # =====================================================
    # Selective-FD: client-side proxy prediction
    # =====================================================
    def get_proxy_predictions(self, proxy_dataset):

        device = self.config.DEVICE
        loader = DataLoader(
            proxy_dataset,
            batch_size=getattr(self.config, "PROXY_BATCH_SIZE", 128),
            shuffle=False
        )

        self.model.eval()

        all_probs = []
        all_mask = []

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)

                logits = self.model(x)
                probs = F.softmax(logits, dim=1)

                conf, _ = probs.max(dim=1)
                mask = conf >= getattr(self.config, "SELECTIVE_TAU_CLIENT", 0.60)

                all_probs.append(probs.cpu())
                all_mask.append(mask.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_mask = torch.cat(all_mask, dim=0)

        return all_probs, all_mask

    # =====================================================
    # Selective-FD: local KD on filtered proxy teacher
    # =====================================================
    def distill_on_proxy(self, proxy_dataset, teacher_probs, valid_mask):

        device = self.config.DEVICE

        if teacher_probs is None or valid_mask is None:
            return

        loader = DataLoader(
            proxy_dataset,
            batch_size=getattr(self.config, "PROXY_BATCH_SIZE", 128),
            shuffle=False
        )

        self.model.train()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LR
        )

        temperature = getattr(self.config, "TEMPERATURE", 2.0)
        kd_weight = getattr(self.config, "SELECTIVE_KD_WEIGHT", 1.0)
        use_soft = getattr(self.config, "SELECTIVE_USE_SOFT", True)
        distill_epochs = getattr(self.config, "SELECTIVE_DISTILL_EPOCHS", 1)

        teacher_probs = teacher_probs.to(device)
        valid_mask = valid_mask.to(device)

        for _ in range(distill_epochs):
            start = 0

            for x, _ in loader:
                bsz = x.size(0)
                end = start + bsz

                batch_teacher = teacher_probs[start:end]
                batch_valid = valid_mask[start:end]

                start = end

                if batch_valid.sum().item() == 0:
                    continue

                x = x.to(device)

                optimizer.zero_grad()
                student_logits = self.model(x)

                if use_soft:
                    loss = F.kl_div(
                        F.log_softmax(student_logits / temperature, dim=1),
                        batch_teacher,
                        reduction="none"
                    ).sum(dim=1)

                    loss = loss[batch_valid].mean() * (temperature ** 2)
                else:
                    hard_targets = torch.argmax(batch_teacher, dim=1)
                    loss = F.cross_entropy(
                        student_logits[batch_valid],
                        hard_targets[batch_valid]
                    )

                loss = kd_weight * loss
                loss.backward()
                optimizer.step()

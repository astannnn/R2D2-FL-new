import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Server:

    def __init__(self, global_model):
        self.global_model = global_model

    # =====================================================
    # FedAvg aggregation
    # =====================================================

    def aggregate(self, client_weights, client_sizes):

        total = float(sum(client_sizes))
        new_weights = copy.deepcopy(client_weights[0])

        for key in new_weights.keys():
            new_weights[key] = client_weights[0][key] * (client_sizes[0] / total)

            for i in range(1, len(client_weights)):
                new_weights[key] += (
                    client_weights[i][key] * (client_sizes[i] / total)
                )

        self.global_model.load_state_dict(new_weights)

    # =====================================================
    # Proxy Distillation
    # =====================================================

    def distill(self, client_models, proxy_dataset, config, print_stats=True):

        if proxy_dataset is None:
            return

        device = config.DEVICE
        temperature = config.TEMPERATURE

        loader = DataLoader(proxy_dataset, batch_size=128, shuffle=False)

        num_clients = len(client_models)
        if num_clients == 0:
            return

        for m in client_models:
            m.eval()

        # =====================================================
        # Reliability estimation
        # =====================================================

        total_samples = 0
        agreement_total = torch.zeros(num_clients, device=device)

        num_classes = None
        agree_by_class = None
        total_by_class = None

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)

                preds_list = []

                for m in client_models:
                    logits = m(x)

                    if num_classes is None:
                        num_classes = logits.size(1)

                        agree_by_class = torch.zeros(
                            (num_clients, num_classes), device=device
                        )
                        total_by_class = torch.zeros(
                            num_classes, device=device
                        )

                    preds_list.append(torch.argmax(logits, dim=1))

                preds = torch.stack(preds_list, dim=0)
                maj, _ = torch.mode(preds, dim=0)

                B = x.size(0)
                total_samples += B

                for k in range(num_clients):
                    agreement_total[k] += (preds[k] == maj).sum()

                for c in range(num_classes):
                    mask = maj == c
                    cnt = mask.sum()

                    if cnt.item() == 0:
                        continue

                    total_by_class[c] += cnt

                    for k in range(num_clients):
                        agree_by_class[k, c] += (
                            ((preds[k] == maj) & mask).sum()
                        )

        r_k = agreement_total / max(1, total_samples)

        r_kc = torch.zeros((num_clients, num_classes), device=device)

        for c in range(num_classes):
            denom = float(total_by_class[c].item())

            if denom <= 0:
                r_kc[:, c] = r_k
            else:
                r_kc[:, c] = agree_by_class[:, c] / denom

        if print_stats:
            print("r_k:", [round(float(x), 4) for x in r_k])

        rk = r_k.clamp(0.05, 0.95)
        rkc = r_kc.clamp(0.05, 0.95)

        # =====================================================
        # Distillation
        # =====================================================

        self.global_model.train()

        optimizer = torch.optim.SGD(
            self.global_model.parameters(),
            lr=getattr(config, "DISTILL_LR", 0.001),
            momentum=0.9
        )

        eps = 1e-12
        logC = math.log(num_classes)

        for x, _ in loader:
            x = x.to(device)

            logits_list = []
            probs_list = []
            preds_list = []

            with torch.no_grad():
                for m in client_models:
                    z = m(x)
                    p = F.softmax(z, dim=1)

                    logits_list.append(z)
                    probs_list.append(p)
                    preds_list.append(torch.argmax(z, dim=1))

                preds = torch.stack(preds_list, dim=0)
                maj, _ = torch.mode(preds, dim=0)

            alphas = torch.zeros(
                (num_clients, x.size(0)),
                device=device
            )

            for k in range(num_clients):
                p_k = probs_list[k]

                ent = -(p_k * torch.log(p_k + eps)).sum(dim=1)
                ent_norm = torch.clamp(ent / logC, 0.0, 1.0)

                if getattr(config, "USE_RELIABILITY", True):
                    if getattr(config, "USE_CLASS_RELIABILITY", True):
                        class_gate = rkc[k, maj]
                    else:
                        class_gate = 1.0

                    alphas[k] = rk[k] * class_gate * (1.0 - ent_norm)
                else:
                    alphas[k] = torch.ones_like(ent_norm)

            alpha_sum = alphas.sum(dim=0).clamp_min(1e-12)
            alphas = alphas / alpha_sum.unsqueeze(0)

            teacher_probs = torch.zeros(
                (x.size(0), num_classes),
                device=device
            )

            for k in range(num_clients):
                p_k = F.softmax(logits_list[k] / temperature, dim=1)
                teacher_probs += alphas[k].unsqueeze(1) * p_k

            teacher_probs = teacher_probs.clamp_min(1e-12)
            teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)

            student_logits = self.global_model(x)

            logp_student = F.log_softmax(
                student_logits / temperature,
                dim=1
            )

            loss_kd = F.kl_div(
                logp_student,
                teacher_probs,
                reduction="batchmean"
            ) * (temperature ** 2)

            loss = config.BETA * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

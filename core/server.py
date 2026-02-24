import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Server:
    def __init__(self, global_model):
        self.global_model = global_model

    # -------------------------------------------------
    # Weighted FedAvg aggregation
    # -------------------------------------------------
    print("Aggregate signature:", "weighted")
    def aggregate(self, client_weights, client_sizes):
        total = float(sum(client_sizes))
        new_weights = copy.deepcopy(client_weights[0])

        for key in new_weights.keys():
            new_weights[key] = client_weights[0][key] * (client_sizes[0] / total)
            for i in range(1, len(client_weights)):
                new_weights[key] += client_weights[i][key] * (
                    client_sizes[i] / total
                )

        self.global_model.load_state_dict(new_weights)

    # -------------------------------------------------
    # R2D2-FL Distillation
    # -------------------------------------------------
    def distill(self, client_models, proxy_dataset, config, print_stats=True):

        device = config.DEVICE
        temperature = config.TEMPERATURE
        loader = DataLoader(proxy_dataset, batch_size=128, shuffle=False)

        num_clients = len(client_models)
        if num_clients == 0:
            return

        for m in client_models:
            m.eval()

        total_samples = 0
        agreement_total = torch.zeros(num_clients)
        num_classes = None

        agree_by_class = None
        total_by_class = None

        # -------------------------------------------------
        # Reliability estimation
        # -------------------------------------------------
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)

                preds_list = []
                for m in client_models:
                    logits = m(x)
                    if num_classes is None:
                        num_classes = logits.size(1)
                        agree_by_class = torch.zeros(
                            (num_clients, num_classes)
                        )
                        total_by_class = torch.zeros(num_classes)
                    preds_list.append(torch.argmax(logits, dim=1))

                preds = torch.stack(preds_list, dim=0)
                maj, _ = torch.mode(preds, dim=0)

                B = x.size(0)
                total_samples += B

                for k in range(num_clients):
                    agreement_total[k] += (preds[k] == maj).sum().item()

                for c in range(num_classes):
                    mask = maj == c
                    cnt = mask.sum().item()
                    if cnt == 0:
                        continue
                    total_by_class[c] += cnt
                    for k in range(num_clients):
                        agree_by_class[k, c] += (
                            ((preds[k] == maj) & mask).sum().item()
                        )

        r_k = agreement_total / max(1, total_samples)

        r_kc = torch.zeros((num_clients, num_classes))
        for c in range(num_classes):
            denom = float(total_by_class[c])
            if denom <= 0:
                r_kc[:, c] = r_k
            else:
                r_kc[:, c] = agree_by_class[:, c] / denom

        if print_stats:
            print("r_k:", [round(float(x), 4) for x in r_k])

        sig_rk = torch.clamp(torch.sigmoid(r_k), 0.2, 0.8)
        sig_rkc = torch.clamp(torch.sigmoid(r_kc), 0.2, 0.8)

        device = next(self.global_model.parameters()).device
        sig_rk = sig_rk.to(device)
        sig_rkc = sig_rkc.to(device)

        # -------------------------------------------------
        # Proxy Distillation
        # -------------------------------------------------
        self.global_model.train()
        optimizer = torch.optim.SGD(
            self.global_model.parameters(), lr=config.LR
        )

        logC = math.log(num_classes)
        eps = 1e-12

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

            teacher_logits = torch.zeros(
                (x.size(0), num_classes), device=device
            )

            alphas = torch.zeros(
                (num_clients, x.size(0)), device=device
            )

            for k in range(num_clients):
                z_k = logits_list[k]
                p_k = probs_list[k]

                ent = -(p_k * torch.log(p_k + eps)).sum(dim=1)
                ent_norm = torch.clamp(ent / logC, 0.0, 1.0)

                class_gate = sig_rkc[k, maj]

                alphas[k] = (
                    sig_rk[k]
                    * class_gate
                    * (1.0 - ent_norm)
                )

            alpha_sum = alphas.sum(dim=0, keepdim=True).clamp_min(1e-12)
            alphas = alphas / alpha_sum

            for k in range(num_clients):
                teacher_logits += (
                    alphas[k].unsqueeze(1) * logits_list[k]
                )

            student_logits = self.global_model(x)

            p_teacher = (
                F.softmax(teacher_logits / temperature, dim=1)
                .clamp_min(1e-12)
            )
            p_teacher = p_teacher / p_teacher.sum(
                dim=1, keepdim=True
            )

            logp_student = F.log_softmax(
                student_logits / temperature, dim=1
            )

            loss_kd = (
                F.kl_div(logp_student, p_teacher, reduction="batchmean")
                * (temperature ** 2)
            )

            loss = config.BETA * loss_kd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
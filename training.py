import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import BATCH_SIZE, LR, TEMPERATURE, PHASE_MIN_DELTA, PHASE_PATIENCE
from data import make_loader
from model import build_model


def train_local(model, dataset, device, epochs, early_stop=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, stale = float("inf"), 0
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for x, y in make_loader(dataset, True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        if early_stop:
            mean_loss = total / max(count, 1)
            if best_loss - mean_loss > PHASE_MIN_DELTA:
                best_loss, stale = mean_loss, 0
            else:
                stale += 1
                if stale >= PHASE_PATIENCE:
                    break


def soft_labels(model, dataset, device):
    outputs = []
    model.eval()
    with torch.no_grad():
        for x, _ in make_loader(dataset, False):
            outputs.append(model(x.to(device)).clamp(-50, 50).cpu())
    return torch.cat(outputs)


def distillation_loss(student_logits, teacher_logits):
    log_q = F.log_softmax(student_logits / TEMPERATURE, dim=1)
    p = F.softmax(teacher_logits / TEMPERATURE, dim=1)
    return TEMPERATURE ** 2 * F.kl_div(log_q, p, reduction="batchmean")


def hybrid_distillation_loss(student_logits, labels, teacher_logits, alpha):
    kl = distillation_loss(student_logits, teacher_logits)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kl + (1 - alpha) * ce


def distill_student(student, ref_dataset, teacher_logits, alpha, device, epochs):
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    student.train()
    for _ in range(epochs):
        for indices in torch.split(torch.arange(len(ref_dataset)), BATCH_SIZE):
            x = ref_dataset.tensors[0][indices].to(device)
            labels = ref_dataset.tensors[1][indices].to(device)
            teacher_batch = teacher_logits[indices].to(device)
            optimizer.zero_grad()
            student_logits = student(x).clamp(-50, 50)
            hybrid_distillation_loss(
                student_logits, labels, teacher_batch, alpha
            ).backward()
            optimizer.step()


def aggregate(states, weights):
    return {
        key: sum(weight * state[key] for state, weight in zip(states, weights))
        for key in states[0]
    }


def reverse_distill(model, ref_dataset, global_logits, alpha, device, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for ref_indices in torch.split(torch.arange(len(ref_dataset)), BATCH_SIZE):
            x_ref = ref_dataset.tensors[0][ref_indices].to(device)
            labels = ref_dataset.tensors[1][ref_indices].to(device)
            teacher_batch = global_logits[ref_indices].to(device)
            optimizer.zero_grad()
            student_logits = model(x_ref).clamp(-50, 50)
            hybrid_distillation_loss(
                student_logits, labels, teacher_batch, alpha
            ).backward()
            optimizer.step()


def evaluate(model, test_loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return 100.0 * correct / total


def ensemble_accuracy(models, weights, test_loader, device):
    correct = total = 0
    for model in models:
        model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            probs = sum(
                weight * torch.softmax(model(x.to(device)).clamp(-50, 50) / TEMPERATURE, dim=1)
                for model, weight in zip(models, weights)
            )
            correct += (probs.argmax(1).cpu() == y).sum().item()
            total += len(y)
    return 100.0 * correct / total


def run_mode(mode, phase_states, global_state, private_sets, ref_dataset, test_loader,
             dataset_name, input_shape, weights, seed, device, alpha, rounds, distill_epochs):
    clients = [build_model(dataset_name, input_shape).to(device) for _ in private_sets]
    for client, state in zip(clients, phase_states):
        client.load_state_dict(copy.deepcopy(state))
    global_model = build_model(dataset_name, input_shape).to(device)
    global_model.load_state_dict(copy.deepcopy(global_state))
    rows = []
    server_acc = evaluate(global_model, test_loader, device)
    client_accs = [evaluate(client, test_loader, device) for client in clients]
    rows.append([mode, seed, 0, server_acc, np.mean(client_accs), np.std(client_accs)])

    for round_number in range(1, rounds + 1):
        teacher_outputs = [soft_labels(client, ref_dataset, device) for client in clients]
        if mode in ("c2s_only", "bidirectional"):
            student = build_model(dataset_name, input_shape).to(device)
            student_states = []
            for teacher_q in teacher_outputs:
                student.load_state_dict(copy.deepcopy(global_model.state_dict()))
                distill_student(
                    student, ref_dataset, teacher_q, alpha, device, distill_epochs
                )
                student_states.append(copy.deepcopy(student.state_dict()))
            global_model.load_state_dict(aggregate(student_states, weights))
            server_acc = evaluate(global_model, test_loader, device)
        else:
            server_acc = ensemble_accuracy(clients, weights, test_loader, device)

        if mode == "bidirectional":
            global_logits = soft_labels(global_model, ref_dataset, device)
            for client in clients:
                reverse_distill(
                    client, ref_dataset, global_logits, alpha, device, distill_epochs
                )

        client_accs = [evaluate(client, test_loader, device) for client in clients]
        rows.append([mode, seed, round_number, server_acc, np.mean(client_accs), np.std(client_accs)])
    return rows

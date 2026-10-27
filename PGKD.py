import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(logits_student, logits_teacher, temperature, reduce=True):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    if reduce:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    else:
        loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    loss_kd *= temperature ** 2
    return loss_kd


def PGKD_loss(logits_student, logits_teacher, tik_factor):
    _, num_classes = logits_student.size()

    logits_student = F.normalize(logits_student, p=2, dim=1)
    logits_teacher = F.normalize(logits_teacher, p=2, dim=1)

    student_matrix = torch.mm(logits_student, logits_student.T) / (num_classes)
    teacher_matrix = torch.mm(logits_teacher, logits_teacher.T) / (num_classes)

    diff = student_matrix - teacher_matrix

    cov_matrix = torch.cov(diff.T) + tik_factor * torch.eye(diff.size(1), device=diff.device)

    L = torch.linalg.cholesky(cov_matrix)
    cov_matrix_inv = torch.cholesky_inverse(L)

    mahalanobis_dist = torch.einsum('bi,ij,bj->b', diff, cov_matrix_inv, diff)
    return torch.sqrt(mahalanobis_dist).mean()


class PGKD(nn.Module):
    def __init__(self, student, teacher, temp=1.0, tik=1e-1, ce=0.0, kd=1.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temp
        self.tik = tik
        self.ce_loss_weight = ce
        self.kd_loss_weight = kd

    def forward_train(self, image, target, epoch):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_node = self.kd_loss_weight * kd_loss(
            logits_student,
            logits_teacher,
            self.temperature
        )

        loss_rel = self.kd_loss_weight * PGKD_loss(
            logits_student,
            logits_teacher,
            self.tik
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_node": loss_node,
            "loss_rel": loss_rel,
        }
        return logits_student, logits_teacher, losses_dict

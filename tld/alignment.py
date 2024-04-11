from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class StudentTeacherPair:
    """Defines the alignment between teacher and student features.
    """
    student_idx : int
    student_dim : int

    teacher_idx : int
    teacher_dim : int

class InterpolateAlignment(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return F.interpolate(x, size = self.out_features, mode = "bilinear", align_corners = True)

class AlignmentLoss(nn.Module):

    def __init__(
            self,
            alignment_map : List[StudentTeacherPair],
            loss_fn : nn.Module = nn.MSELoss(),
            alignment_method = "linear" # One of "linear" or "interpolate"
        ):
        super().__init__()

        self.loss_fn = loss_fn
        self.alignment_map = alignment_map
        self.student_to_teacher_indices = {pair.student_idx : pair.teacher_idx for pair in alignment_map}
        assert len(self.student_to_teacher_indices) == len(alignment_map), "student indices must be unique"
        assert len(set(self.student_to_teacher_indices.values())) == len(self.student_to_teacher_indices), "teacher indices must be unique"

        if alignment_method == "linear":
            projection_module = nn.Linear
        elif alignment_method == "interpolate":
            projection_module = InterpolateAlignment
        else:
            raise ValueError(f"Invalid alignment method: {alignment_method}")

        self.student_to_teacher_projections = nn.ModuleDict(
            {
                # Remark: ModuleDict keys must be strings
                str(pair.student_idx) :
                projection_module(in_features = pair.student_dim, out_features = pair.teacher_dim)
                for pair in alignment_map
            }
        )
        
    def forward(self, student_features, teacher_features):
        """
        Parameters
        ----------
        student_features : list
            List of torch tensors of shape (bs, d)
        teacher_features : list
            List of torch tensors of shape (bs, d)
        """

        loss = 0

        for i, student_feature in enumerate(student_features):
            teacher_idx = self.student_to_teacher_indices.get(i, None)
            if teacher_idx is None:
                continue

            loss += self.loss_fn(
                teacher_features[teacher_idx],
                self.student_to_teacher_projections[str(i)](student_feature)
            )

        return loss



        
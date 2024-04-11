from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

class AlignmentNetwork(nn.Module):

    def __init__(
            self, 
            alignment_map : Dict[int, int],
        ):
        super().__init__()

        self.alignment_map = alignment_map
        self.teacher_to_student_projections = nn.ModuleDict(
            {  
                # Remark: ModuleDict keys must be strings
                str(teacher_feature_index) :
                nn.Linear(in_features = teacher_dim, out_features = student_dim)
                for teacher_feature_index, (teacher_dim, student_dim) in alignment_map.items()
            }
        )
        
    def forward(self, teacher_features):
        """
        Parameters
        ----------
        teacher_features : list
            List of torch tensors of shape (bs, d)
        """
        student_features = []
        for i, teacher_feature in enumerate(teacher_features):
            key = str(i)
            if key not in self.teacher_to_student_projections:
                continue
            student_features.append(
                self.teacher_to_student_projections[key](teacher_feature)
            )
        
        return student_features



        
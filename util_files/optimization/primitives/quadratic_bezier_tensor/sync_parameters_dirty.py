import torch


def synchronize_parameters(self):
    r"""Synchronize the auxiliary parameters of the curve tensor with the main, optimized parameters.
    Call each time the main parameters are updated.

    Notes
    -----
    The algorithm is laced with the assumption that the changes to the main parameters are small.
    It is simple and dirty, so let's see if it works.

    1. Take old P2 and directions P2 -> P1 and P2 -> P3, new lengths P2 -> P1 and P2 -> P3, and update P1 and P3.
    2. Calculate tb from P1, P2, P3.
       Since P1 and P3 moved only slightly along the tangent directions at the ends of the curve,
       we assume that step 1 doesn't result in change in the position of B.
    3. So, calculate intermediate B_int from tb from step 2 and move P1, P2, P3 as B_int -> new B.
    4. Calculate lengths B -> P1 and B -> P3.
    5. Update directions B -> P1 and B -> P3 in accordance to their updates by optimizer.
       As the result, recalculate P1 and P3 from B, new directions B -> P1 and B -> P3, and just calculated lengths.
       Since changes are small, we assume that inconsistency
       between the meanings of the old and new directions B -> P1 won't mess everything up.
    6. Find new curve that goes from P1 through B to P3.
       Since changes are small, we assume that after the step 5,
       the ratio of the lengths of the arcs P1 -> B and B -> P3 of the new curve are roughly the same
       as on the old curve, and thereby the values of t for B on the curves are the same.
       So, P2 for the new curve is given by
       P1 * (1 - tb)^2 + P2 * (1 - tb) * tb * 2 + P3 * tb^2 = B.
    7. Update the main and auxiliary parameters w.r.t these P1, P2, P3
    """
    # 1. Take old P2 and directions P2 -> P1 and P2 -> P3,
    p2 = self.p2_aux.data
    p2_to_p1 = self.p2_to_p1.data
    p2_to_p3 = self.p2_to_p3.data
    #    new lengths P2 -> P1 and P2 -> P3,
    p2_to_p1_len = self._p2_to_p1_len.data
    p2_to_p3_len = self._p2_to_p3_len.data
    #    and update P1 and P3.
    p1 = p2 + p2_to_p1 * p2_to_p1_len
    p3 = p2 + p2_to_p3 * p2_to_p3_len

    # 2. Calculate tb from P1, P2, P3.
    _ = torch.sqrt(p2_to_p1_len)
    tb = _ / (_ + torch.sqrt(p2_to_p3_len))

    # 3. So, calculate intermediate B_int from tb from step 2
    b = p1 * (1 - tb).pow(2) + p2 * (1 - tb) * tb * 2 + p3 * tb.pow(2)
    #    and move P1, P2, P3 as B_int -> new B.
    db = self._b.data - b
    p1 = p1 + db
    p2 = p2 + db
    p3 = p3 + db
    b = p1 * (1 - tb).pow(2) + p2 * (1 - tb) * tb * 2 + p3 * tb.pow(2)

    # 4. Calculate lengths B -> P1 and B -> P3.
    b_to_p1 = p1 - b
    b_to_p1_len = torch.norm(b_to_p1, dim=1, keepdim=True)
    theta1 = torch.atan2(b_to_p1[:, 1], b_to_p1[:, 0]).unsqueeze(1)

    b_to_p3 = p3 - b
    b_to_p3_len = torch.norm(b_to_p3, dim=1, keepdim=True)
    theta2 = torch.atan2(b_to_p3[:, 1], b_to_p3[:, 0]).unsqueeze(1)

    # 5. Update directions B -> P1 and B -> P3 in accordance to their updates by optimizer.
    theta1 += self._theta1.data - self.theta1_stored.data
    theta2 += self._theta2.data - self.theta2_stored.data
    #    As the result, recalculate P1 and P3
    b_to_p1 = torch.cat([torch.cos(theta1), torch.sin(theta1)], dim=1)
    p1 = b + b_to_p1 * b_to_p1_len
    b_to_p3 = torch.cat([torch.cos(theta2), torch.sin(theta2)], dim=1)
    p3 = b + b_to_p3 * b_to_p3_len

    # 6. Find new curve that goes from P1 through B to P3.
    p2 = ((b - p3 * tb.pow(2) - p1 * (1 - tb).pow(2)) / (tb * (1 - tb) * 2))

    # 7. Update the main and auxiliary parameters w.r.t these P1, P2, P3
    self.set_parameters(p1, p2, p3)

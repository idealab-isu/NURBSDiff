import numpy as np


def Surf_pt(u, v, P, knot_u, knot_v, degree_u, degree_v):
    count = knot_v.shape[0] - degree_v - 1
    span_u = span_linear(knot_u.shape[0] - degree_u - 1, knot_u, u)
    span_v = span_linear(knot_v.shape[0] - degree_v - 1, knot_v, v)

    N_u = Basis_Surf(u, degree_u, span_u, knot_u)
    N_v = Basis_Surf(v, degree_v, span_v, knot_v)

    S = np.zeros(3)

    uind = span_u - degree_u
    for i in range(0, degree_v + 1):
        temp = np.zeros(3)
        vind = span_v - degree_v + i

        for j in range(0, degree_u + 1):
            temp[0] = temp[0] + N_u[j] * P[((uind + j) * count) + vind][0]
            temp[1] = temp[1] + N_u[j] * P[((uind + j) * count) + vind][1]
            temp[2] = temp[2] + N_u[j] * P[((uind + j) * count) + vind][2]

        S[0] = S[0] + N_v[i] * temp[0]
        S[1] = S[1] + N_v[i] * temp[1]
        S[2] = S[2] + N_v[i] * temp[2]

    return S


# A3.6
def Deri_Surf(u, v, order, P, knot_u, knot_v, degree_u, degree_v):
    d = np.array([min(3, order), min(3, order)])

    count = knot_v.shape[0] - degree_v - 1
    span_u = span_linear(knot_u.shape[0] - degree_u - 1, knot_u, u)
    span_v = span_linear(knot_v.shape[0] - degree_v - 1, knot_v, v)

    SKL = np.zeros([2, 2, 3])

    ders_u = Basis_Deri(u, order, span_u, degree_u, knot_u)
    ders_v = Basis_Deri(v, order, span_v, degree_v, knot_v)

    for k in range(0, d[0] + 1):
        temp = np.zeros([4, 3])
        for s in range(0, degree_v + 1):
            for r in range(0, degree_u + 1):
                temp[s][0] = temp[s][0] + ders_u[k][r] * P[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    0]
                temp[s][1] = temp[s][1] + ders_u[k][r] * P[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    1]
                temp[s][2] = temp[s][2] + ders_u[k][r] * P[((span_u - degree_u + r) * count) + (span_v - degree_v + s)][
                    2]

        dd = min(order - k, d[1])
        for l in range(0, dd + 1):
            # SKL[(k * 3) + l][0] = 0.0 ; SKL[(k * 3) + l][1] = 0.0 ; SKL[(k * 3) + l][2] = 0.0
            for s in range(0, degree_v + 1):
                SKL[k][l][0] = SKL[k][l][0] + (ders_v[l][s] * temp[s][0])
                SKL[k][l][1] = SKL[k][l][1] + (ders_v[l][s] * temp[s][1])
                SKL[k][l][2] = SKL[k][l][2] + (ders_v[l][s] * temp[s][2])

    return SKL

def span_linear(CNTRL_PTS_Count, knot_vec, knot):
    span_lin = 0

    while span_lin < CNTRL_PTS_Count and knot_vec[span_lin] <= knot:
        span_lin += 1

    return span_lin - 1


def Basis_Surf(u, degree, span_i, knotV):
    left = np.empty([4])
    right = np.empty([4])

    N = np.empty([4])

    N[0] = 1.0
    for j in range(1, degree + 1):
        left[j] = u - knotV[span_i + 1 - j]
        right[j] = knotV[span_i + j] - u
        saved = 0.0

        for r in range(0, j):
            temp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp

        N[j] = saved

    return N


def Basis_Deri(u, order, span, degree, knotV):
    left = np.empty([4], dtype=np.float32)
    right = np.empty([4], dtype=np.float32)
    ndu = np.full([4, 4], 1.0)  # ndu[0][0] = 1.0
    ders = np.zeros([2, 4])

    for j in range(1, degree + 1):
        left[j] = u - knotV[span + 1 - j]
        right[j] = knotV[span + j] - u
        saved = 0.0

        for r in range(0, j):
            ndu[j][r] = right[r + 1] + left[j - r]
            temp = ndu[r][j - 1] / ndu[j][r]

            ndu[r][j] = saved + (right[r + 1] * temp)
            saved = left[j - r] * temp

        ndu[j][j] = saved

    for j in range(0, degree + 1):
        ders[0][j] = ndu[j][degree]

    a = np.full([4, 2], 1.0)

    for r in range(0, degree + 1):
        s1 = 0
        s2 = 1
        a[0][0] = 1.0

        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
                d = a[s2][0] * ndu[rk][pk]

            if rk >= -1:
                j1 = 1
            else:
                j1 = -rk
            if (r - 1) <= pk:
                j2 = k - 1
            else:
                j2 = degree - r

            for j in range(j1, j2 + 1):
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
                d += (a[s2][j] * ndu[rk + j][pk])

            if r <= pk:
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
                d += (a[s2][k] * ndu[r][pk])
            ders[k][r] = d

            # Switch rows
            j = s1
            s1 = s2
            s2 = j

    r = degree
    for k in range(1, order + 1):
        for j in range(0, degree + 1):
            ders[k][j] *= r
        r *= (degree - k)

    return ders


def compute_surf_offset(off_layers, CNTRL_PTS, knot_u, knot_v, degree_u, degree_v):
    delta_u = 64
    delta_v = 64
    grid_1, grid_2 = np.meshgrid(np.linspace(0.0, 1.0, delta_u), np.linspace(0.0, 1.0, delta_v))
    OFF_PTS = np.empty([off_layers, grid_1.shape[0], grid_1.shape[1], 3], dtype=np.float32)
    thickness = 5.0
    for i in range(0, grid_1.shape[0]):
        for j in range(0, grid_1.shape[1]):

            # PTS, normals = compute_normal_surface(CNTRL_PTS, knot_u, knot_v, grid_1, grid_2)
            PT = Surf_pt(grid_1[i][j], grid_2[i][j], CNTRL_PTS, knot_u, knot_v, degree_u, degree_v)
            deri = Deri_Surf(grid_1[i][j], grid_2[i][j], 1, CNTRL_PTS, knot_u, knot_v, degree_u, degree_v)
            temp = np.cross(deri[0][1], deri[1][0])
            normals = temp / np.linalg.norm(temp)

            # if PT[1] < 12:
            #     thickness = 0.5
            # elif 40 > PT[1] > 12:
            #     thickness = 0.25
            # else:
            #     thickness = 0.1

            for layer in range(0, off_layers):
                OFF_PTS[layer][i][j] = PT + (thickness * layer * normals)

    return OFF_PTS
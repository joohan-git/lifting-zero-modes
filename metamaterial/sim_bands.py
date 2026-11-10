"""Band structures of the beam-frame composite along a closed generic path, for the three joint designs.

High-symmetry lines lie on the sets where the floppy net or the scaffold gains mechanisms, so the path joins four
generic wavevectors (reduced coordinates of the conventional cell) and the floppy-net gap g(q) is stored with it.
k_c = 1e-3, k_b = k_c, thick cage (10, 1), kf_a = 1, k_cr = 100. Run: python metamaterial/sim_bands.py --rcsr DIR   (stored result: metamaterial/bands.json)
"""
import json
import numpy as np
import sim_core as S

P = np.array([[0.06, 0.13, 0.21], [0.44, 0.17, 0.29], [0.38, 0.41, 0.09], [0.11, 0.36, 0.45], [0.06, 0.13, 0.21]])
NSEG = 60
PAIRS = [("bcu", "nbo"), ("nia", "nia-d"), ("pcu", "urk"), ("hex", "bnn"), ("fcu", "flu")]
KC = 1e-3


def path():
    qs, x = [], []; acc = 0.0
    for a, b in zip(P[:-1], P[1:]):
        for t in np.linspace(0, 1, NSEG, endpoint=False):
            qs.append(a + t * (b - a)); x.append(acc + t * np.linalg.norm(b - a))
        acc += np.linalg.norm(b - a)
    qs.append(P[-1]); x.append(acc)
    return np.array(qs), np.array(x), [float(sum(np.linalg.norm(P[k + 1] - P[k]) for k in range(i))) for i in range(len(P))]


def main():
    qs, x, ticks = path(); out = dict(path=P.tolist(), x=x.tolist(), ticks=ticks, kc=KC, kb=KC, pairs={})
    for sc, fl in PAIRS:
        m = S.Model(sc, fl); rec = dict(M=m.M, gap=[m.gap(q) for q in qs])
        for des in ('rigid', 'lock', 'release'):
            W, Wt, lift = [], [], []
            for q in qs:
                lw, lwt, w, v = m.lifted(q, kc=KC, kf_a=1.0, kf_b=KC, ks_a=10.0, ks_b=1.0, design=des, kcr=100.0)
                Z = np.zeros((3 * m.n, m.M), complex); Zq = m.Z(q); Z[:3 * m.nf, :min(m.M, Zq.shape[1])] = Zq[:, :m.M]
                wt = np.sum(np.abs(Z.conj().T @ v) ** 2, axis=0)
                keep = w / KC < 12
                W.append((w[keep] / KC).tolist()); Wt.append(wt[keep].tolist()); lift.append((lw / KC).tolist())
            rec[des] = dict(w2=W, weight=Wt, lifted=lift)
        out['pairs'][f'{sc}|{fl}'] = rec
        print(sc, fl, flush=True)
    json.dump(out, open(S.out('bands.json'), 'w'))


if __name__ == '__main__':
    main()

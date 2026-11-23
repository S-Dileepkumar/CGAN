from .synthetic import copy as _copy, induction as _induction, associative_recall as _ar

SYNTHETIC = {
    "copy": _copy.make_batch,
    "induction": _induction.make_batch,
    "assoc_recall": _ar.make_batch,
}

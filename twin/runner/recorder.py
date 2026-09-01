"""에피소드 로깅 — npz + meta.json.

parquet 이 아니라 npz 인 이유는 ``pyarrow`` 가 이 환경에 없어서다. 배열이 균질하고
한 에피소드가 수천 스텝이라 npz 로 충분하다 (pandas 로 바로 읽힌다).

메타를 같이 남기는 이유
-----------------------
숫자만 남기면 **반년 뒤에 무엇을 잰 것인지 알 수 없다.** 이 저장소는 이미
"어느 판을 어느 규약으로 평가했는지" 때문에 결론이 뒤집힌 적이 있다 (v29 vs v30 의
요 보정). 그래서 체크포인트, 관측 md5, 요 보정, 계획 경로, 맵, ff_mode 를 함께 박는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np


def md5(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class Recorder:
    """스텝마다 ``add(**kv)``, 끝나면 ``save(dir)``."""

    def __init__(self, meta: dict | None = None):
        self._cols: dict[str, list] = defaultdict(list)
        self.meta = dict(meta or {})
        self._n = 0

    def add(self, **kv) -> None:
        for k, v in kv.items():
            self._cols[k].append(np.asarray(v, dtype=np.float64))
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def array(self, key: str) -> np.ndarray:
        return np.stack(self._cols[key]) if key in self._cols else np.empty(0)

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self._cols.items()}
        path = os.path.join(out_dir, "log.npz")
        np.savez_compressed(path, **arrays)
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump({**self.meta,
                       "n_steps": self._n,
                       "columns": {k: list(v.shape[1:]) for k, v in arrays.items()}},
                      fh, indent=2, ensure_ascii=False)
        return path

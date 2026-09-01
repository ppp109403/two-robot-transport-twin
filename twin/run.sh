#!/usr/bin/env bash
# Isaac 계열 스크립트 실행 래퍼.
#
# 두 가지를 동시에 해야 한다 (둘 중 뭘 빠뜨렸는지는 에러 메시지로 구분된다):
#
#   conda activate <env>     activate.d 훅이 _isaac_sim/setup_conda_env.sh 를 source 해서
#                            omni/isaacsim 경로를 PYTHONPATH 에 넣는다.
#                            빠뜨리면 -> ModuleNotFoundError: isaacsim
#   PATH 강제                다른 venv 가 활성화돼 있으면 python 을 가로챈다. 비대화형
#                            셸에서는 deactivate 가 함수로 안 잡혀 해제되지 않는다.
#                            빠뜨리면 -> ModuleNotFoundError: rsl_rl
#
# ★ set -u 를 켜지 말 것 — setup_conda_env.sh 가 $ZSH_VERSION 을 언바운드로 참조해서
#   conda activate 가 그 자리에서 죽는다 ("ZSH_VERSION: unbound variable").
#
# 다른 기계에서 쓸 때 바꿀 것은 두 개뿐이고, 환경변수로 덮을 수 있다::
#
#     TWIN_CONDA_ROOT   conda 설치 위치      (기본: $HOME/miniconda3)
#     TWIN_CONDA_ENV    Isaac Lab conda env  (기본: ppo_afl)
set -eo pipefail

CONDA_ROOT="${TWIN_CONDA_ROOT:-$HOME/miniconda3}"
CONDA_ENV="${TWIN_CONDA_ENV:-ppo_afl}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "conda 가 없다: $CONDA_ROOT" >&2
    echo "TWIN_CONDA_ROOT 로 위치를 지정할 것." >&2
    exit 1
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PATH="$CONDA_ROOT/envs/$CONDA_ENV/bin:$PATH"
hash -r
cd "$REPO"

exec python -u "$@"

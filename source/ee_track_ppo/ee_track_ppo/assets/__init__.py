# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""이 프로젝트에서 직접 들여온 로봇 에셋 모음.

Isaac Lab 기본 제공(``isaaclab_assets``)에 없는 로봇만 여기에 둔다.

- ``fairino_fr3`` : Fairino FR3 (fairino3_v6). 공식 ROS2 저장소의 URDF 를 변환해서 사용.
"""

import os

# 이 패키지에 딸린 원본 데이터(urdf/meshes/usd) 루트
ASSETS_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
"""원본 에셋 데이터 디렉터리 (``.../ee_track_ppo/assets/data``)."""

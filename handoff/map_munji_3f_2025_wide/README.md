# 맵 이관 — `munji_3f_2025_wide`

2026-08-05. 관제 PC(학습 PC) 플래너가 **실제로 로드해서 계획을 푼** 맵이다.

```
munji_3f_2025_wide.yaml   md5 c0fc7b19516a5f0b9981bb4b3f1225ea
munji_3f_2025_wide.pgm    md5 947731dec6b13d7d5548628e169d6845   1.3 MB
```

## 왜 보내나

로봇 PC 보고에 따르면 그쪽 q4 기본값은 ``munji_3f_wide/munji_3f_2025_wide.yaml`` 인데,
**이 PC 에는 ``munji_3f_wide/`` 라는 디렉토리가 없다.** 있는 것은
``munji_3f_2025_wide/`` 뿐이다. 디렉토리가 다르니 다른 파일일 가능성이 크다.

맵이 어긋나면 **에러 없이 목표가 엉뚱한 데 찍힌다.** 플래너는 자기 map_server 로,
로봇은 q4 의 map_server 로 각자 로컬 파일을 읽고 ``/map`` 은 도메인 브리지를
넘어가지 않기 때문이다.

## 확정값

```
image        munji_3f_2025_wide.pgm
resolution   0.05
origin       [-46, -31, 0]
size         1151 x 1121
mode         trinary,  occupied 0.65,  free 0.25
```

.. warning::
   yaml 안에 ``# origin: [-1.27, -5.02, 0]`` 이 주석으로 남아 있다. **쓰지 않는 값이다.**
   활성 origin 은 ``[-46, -31, 0]`` 이다.

.. warning::
   **같은 폴더에 ``munji_3f_2025_wide_room`` 이라는 다른 맵이 있다.** 이름이 비슷하고
   origin 이 ``[-1.27, -5.02, 0]`` 으로 완전히 다르다. 헷갈리지 말 것 —
   ``_room`` 이 붙지 않은 쪽이 플래너가 쓰는 맵이다.

## 쓰는 법

로봇 PC 에서 q4 가 읽는 경로에 덮어쓰거나, 경로를 직접 지정한다::

    ros2 launch <q4> map:=<이 폴더>/munji_3f_2025_wide.yaml

**yaml 안의 ``image:`` 는 상대경로**라 pgm 이 같은 폴더에 있어야 한다.

## 확인

양쪽에서 md5 가 같은지 보면 끝난다::

    md5sum munji_3f_2025_wide.yaml munji_3f_2025_wide.pgm

로드 후에는 로그에서 크기를 대조할 수 있다::

    /map 1151x1121 @ 0.050 m

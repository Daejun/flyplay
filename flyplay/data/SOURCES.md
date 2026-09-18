# flyplay/data — 측정 데이터 출처

`scripts/18_build_brain_data.py`가 원본을 `data/raw/`(git 제외)에 받아 이 폴더의 작은 표로 줄입니다.
원본을 다시 받고 표를 다시 만들려면:

```bash
.venv\Scripts\python.exe scripts\18_build_brain_data.py --download
```

| 파일 | 내용 | 출처 | 라이선스 |
|---|---|---|---|
| `door_odorants.csv` | 냄새 물질 12개가 사구체 51개의 수용체를 얼마나 흥분시키는지(DoOR 합의값, 자발 발화 뺀 값, 빈칸 = 측정 없음). 3-옥탄올의 Or13a(DC2) 반응은 시약에 섞인 1-옥텐-3-올 때문이라 뺐습니다(Lüdke 등 2025) | DoOR 2.0.1 — Münch & Galizia (2016) *Sci Rep* 6:21841, [ropensci/DoOR.data](https://github.com/ropensci/DoOR.data) `v2.0.1` | **CC BY-SA 4.0** — 이 표도 같은 조건으로 배포합니다 |
| `hemibrain_kc.npz` | 오른쪽 버섯체 케니언 세포 1927개의 종류와, 사구체마다 받는 투사 뉴런 시냅스 수 | hemibrain v1.2 — Scheffer 등 (2020) *eLife* 9:e57443, [compact connectome](https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz) | CC BY 4.0 |
| (위 표를 만들 때 사용) | 단일 사구체 투사 뉴런의 사구체 이름 | Schlegel 등 (2021) *eLife* 10:e66018, [flyconnectome/hemibrain_olf_data](https://github.com/flyconnectome/hemibrain_olf_data) `FIB_uPNs.csv` | MIT |

모델에서 이 표를 쓰는 곳은 `flyplay/olfactory.py`의 `ConnectomeFrontEnd`입니다.

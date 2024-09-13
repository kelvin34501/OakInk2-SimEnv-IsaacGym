class ShadowHand__NoForearm__Meta:
    ## attr
    body_name_map = {
        # "wrist": "robot0:wrist",
        "palm": "robot0:palm",
        "ff": "robot0:ffdistal",
        "mf": "robot0:mfdistal",
        "rf": "robot0:rfdistal",
        "lf": "robot0:lfdistal",
        "th": "robot0:thdistal",
    }
    rev_body_name_map = {v: k for k, v in body_name_map.items()}
    body_name_list = [
        "palm",
        "ff",
        "mf",
        "rf",
        "lf",
        "th",
    ]
    body_list = [
        "robot0:palm",
        "robot0:ffdistal",
        "robot0:mfdistal",
        "robot0:rfdistal",
        "robot0:lfdistal",
        "robot0:thdistal",
    ]
    fingertip_name_list = [
        "ff",
        "mf",
        "rf",
        "lf",
        "th",
    ]
    fingertip_list = [
        "robot0:ffdistal",
        "robot0:mfdistal",
        "robot0:rfdistal",
        "robot0:lfdistal",
        "robot0:thdistal",
    ]

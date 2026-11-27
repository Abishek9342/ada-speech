# Needs owner decision: MILE prepare + audit (leakage must be 0)

Time: 2026-11-28T00:25:43

The overnight script (C:\ada-data\overnight.ps1) is paused at this step.

- Fix it and push to main: the script pulls within 5 minutes and retries this step.
- Or push logs/OWNER_REPLY.md containing SKIP to skip this step.

Full output of each step: runs\<step>.txt on the laptop.

## Last 40 lines of logs/overnight.log

```n    models/ada-stt-base-ct2                         60.64   15.29   0.13
    models/ada-stt-base-combo-ct2                   26.03    4.77   0.13
2026-11-27T23:09:50  [run2-eval-fleurs] exit 0
    Evaluating on 150 clips from data\stt\fleurs-ta\test (cpu)
    model                                           WER %   CER %    RTF
    base                                           102.51   61.01   0.41
    models/ada-stt-base-ct2                         55.46   18.91   0.35
    models/ada-stt-base-combo-ct2                   38.33   14.44   0.36
2026-11-27T23:44:54  [run2-eval-fleurs-nd] exit 0
    Evaluating on 150 clips from data\stt\fleurs-ta\test (cpu)
    model                                           WER %   CER %    RTF
    base                                            98.55   54.47   0.39
    models/ada-stt-base-ct2                         52.61   16.32   0.35
    models/ada-stt-base-combo-ct2                   33.62   11.35   0.35
2026-11-27T23:44:54  === RUN 2 DONE (ok=True)
2026-11-27T23:44:58  git push exit 0 (Overnight log: run 2 (CV + Kathbath) results)
2026-11-27T23:44:58  MILE archive OK
2026-11-27T23:44:58  deleted corrupt mile .bad file
2026-11-27T23:47:31  MILE extracted (tar exit 0)
2026-11-28T00:25:36  [mile-prep] exit 0
    Filters removed 254 of 77314 train clips.
    Wrote {'train': 77060, 'test': 12087} examples to data\stt\mile-ta
2026-11-28T00:25:43  [mile-audit] exit 0
    == Size ==
    train  clips=77060  hours=133.02  unknown_duration=0  median_s=5.3  speakers=638  top_speaker_share=0.003
    test   clips=12087  hours=16.81  unknown_duration=0  median_s=4.57  speakers=430  top_speaker_share=0.005
    == Train/test leakage ==
    test_sentences_seen_in_train: 0
    test_speakers_seen_in_train: 430
    test_clips: 12087
    == Flags (clip may be counted under several reasons) ==
    [train] 77060 clips, 1206 flags
         1177  transcript repeated Nx in train
           28  longer than Ns
            1  speaking rate N chars/s vs median N
    [test] 12087 clips, 7 flags
            5  very short audio
            2  longer than Ns
    Wrote 1213 flags to data\stt\mile-ta\audit_flags.csv. Listen to a sample before deciding.
2026-11-28T00:25:43    LEAK: test_speakers_seen_in_train: 430
```

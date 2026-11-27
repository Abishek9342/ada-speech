# Resolved by Claude (no owner action needed)

MILE's published test reuses all 430 test speakers from train. Same fix you chose for Kathbath: official test -> test-known, 16 unseen train speakers held out into dev/test, train deduped against all held-out sentences. Code in scripts/stt/prepare_data.py (tests pass). The overnight script retries the MILE step after this push.

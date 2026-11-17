# Two-GPU tensor parallelism
Approved scope: Q/K/V and gate/up row shards, O/down column shards. Replicated normalization, embedding and output head. Each attention/MLP phase sums both partial outputs using the existing synchronous P2P transfer implementation. No PP+TP composition, KV cache or speedup promise. Reject non-divisible head/KV/intermediate sizes and split-layer with TP.

Plan: first fail CLI numerical test; implement shard loader and explicit TP option; run layer phases on both devices before reduction; verify full logits at lengths 1/4/32/256 in both device orders, invalid options, existing CTest/PP/chat. Current default streams permit independent device work; transfers provide source synchronization.

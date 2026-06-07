import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_duck_v2/finetune_log.jsonl"
rows = [json.loads(l) for l in open(path)]
print("updates logged:", len(rows))
for r in rows:
    if r["update"] % 5 == 0 or "eval_mean_score" in r:
        ev = ""
        if "eval_mean_score" in r:
            ev = f" || EVAL score {r['eval_mean_score']:.1f} duck {r['eval_duck_frac']*100:.1f}% jump {r['eval_jump_frac']*100:.1f}%"
        print(f"upd {r['update']:3d} | birdbuf {r.get('bird_buffer_size',0):5d} | bc_loss {r.get('duck_bc_loss',0):.2f} | H {r['entropy']:.2f} | curr_score {r.get('mean_score_curriculum',0):.0f}{ev}")

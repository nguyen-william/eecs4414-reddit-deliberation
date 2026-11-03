import sys

from src.actor_layer import load_comments, build_actor_layer, basic_stats

path = sys.argv[1]
name = sys.argv[2]

comments = load_comments(path)
G, skipped = build_actor_layer(comments)
stats = basic_stats(G)

print(f"--- {name} ---")
print(f"comments: {len(comments)}, threads: {comments['thread_id'].nunique()}")
print("skipped replies:", dict(skipped))
for key, value in stats.items():
    print(f"{key}: {value}")

top = sorted(G.in_degree(weight="weight"), key=lambda x: x[1], reverse=True)[:5]
print("top 5 users by replies received:", [count for _, count in top])

# Query Top-4 vs Full-History SimplerEnv Evaluation

## Setup

- Model checkpoint: `memvla-bridge.pt`
- Evaluation: closed-loop robot-policy execution in SimplerEnv
- Tasks: 4
- Episodes: 24 per task, 96 per method
- Full-history condition: `query_retrieval_mode=off`
- Query condition: `query_retrieval_mode=query`, `query_retrieval_top_k=4`
- Spatial mode: `baseline` for both conditions

## Results

| Task | Full history | Query top-4 | Change |
|---|---:|---:|---:|
| Put carrot on plate | 18/24 (75.0%) | 19/24 (79.2%) | +4.2 points |
| Stack green cube on yellow cube | 9/24 (37.5%) | 8/24 (33.3%) | -4.2 points |
| Put eggplant in basket | 24/24 (100.0%) | 24/24 (100.0%) | No change |
| Put spoon on tablecloth | 18/24 (75.0%) | 19/24 (79.2%) | +4.2 points |
| **Overall** | **69/96 (71.9%)** | **70/96 (72.9%)** | **+1.0 point** |

## Conclusion

Query top-4 matched full-history performance overall and produced one additional success across 96 episodes. It improved the carrot and spoon tasks, reduced cube stacking by one success, and made no difference on eggplant placement.

This is not strong evidence that query retrieval improves task success: the net difference is only one episode and the task-level effect is mixed. The result does show that selecting four memories can retain approximately the same success rate as the full-history condition. Additional seeds or episodes would be needed to establish a reliable accuracy improvement.

This evaluation measured actual robot-task success. It is separate from the RGB-vs-RGB+spatial offline probe, whose grasp-accuracy metric is not directly comparable to SimplerEnv success rate.
